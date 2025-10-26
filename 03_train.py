import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import json
from tqdm import tqdm
import os

# Import our model classes
from model import GATEncoder, LinkPredictor

# --- Configuration ---
DATA_DIR = 'data'
NODES_FILE = os.path.join(DATA_DIR, 'nodes.csv')
EDGES_FILE = os.path.join(DATA_DIR, 'edges.csv')
NODE_FEATURES_FILE = os.path.join(DATA_DIR, 'node_features.pt')
NODE_MAP_FILE = os.path.join(DATA_DIR, 'paper_id_to_node_idx.json')

# Temporal split configuration
TRAIN_YEAR_END = 2017 # Train on all data up to and including this year
VAL_YEAR = 2018      # Validate on data from this year
TEST_YEAR = 2019       # Test on data from this year

# Model Hyperparameters
# Feature dim depends on the model from step 02
# 768 for 'distilbert-base-uncased'
# 384 for 'sentence-transformers/all-MiniLM-L6-v2'
IN_CHANNELS = 768 
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 128
GAT_HEADS = 4
LEARNING_RATE = 0.001
EPOCHS = 50

# Trained model output paths
ENCODER_MODEL_PATH = os.path.join(DATA_DIR, 'gat_encoder.pt')
PREDICTOR_MODEL_PATH = os.path.join(DATA_DIR, 'gat_predictor.pt')
# ---------------------

def load_graph_data():
    """
    Loads all preprocessed data and performs the temporal split.
    """
    print("Loading graph data...")
    
    # 1. Load node features
    x = torch.load(NODE_FEATURES_FILE)
    num_nodes = x.size(0)
    
    # 2. Load node map
    with open(NODE_MAP_FILE, 'r') as f:
        paper_id_to_node_idx = json.load(f)
        
    # 3. Load node metadata (for year)
    nodes_df = pd.read_csv(NODES_FILE)
    node_idx_to_year = pd.Series(nodes_df.year.values, index=nodes_df.node_idx).to_dict()
    
    # 4. Load edges
    edges_df = pd.read_csv(EDGES_FILE)
    
    # 5. Map string IDs in edges_df to integer node_idx
    print("Mapping edge IDs to integer indices...")
    edges_df['source_idx'] = edges_df['source_id'].map(paper_id_to_node_idx)
    edges_df['target_idx'] = edges_df['target_id'].map(paper_id_to_node_idx)
    
    # Drop edges where one of the nodes is not in our map (e.g., filtered out)
    edges_df = edges_df.dropna(subset=['source_idx', 'target_idx'])
    
    # Convert to integer indices
    edges_df['source_idx'] = edges_df['source_idx'].astype(int)
    edges_df['target_idx'] = edges_df['target_idx'].astype(int)
    
    # 6. Get year for each edge's SOURCE node
    edges_df['source_year'] = edges_df['source_idx'].map(node_idx_to_year)
    
    # 7. Perform temporal split
    print("Performing temporal split...")
    
    # Training edges: Used for message passing and training the predictor
    train_edges_df = edges_df[edges_df['source_year'] <= TRAIN_YEAR_END]
    train_edge_index = torch.tensor([train_edges_df['source_idx'].values, 
                                     train_edges_df['target_idx'].values], dtype=torch.long)
                                     
    # Validation edges: Used for validating the link predictor
    val_edges_df = edges_df[edges_df['source_year'] == VAL_YEAR]
    val_edge_index = torch.tensor([val_edges_df['source_idx'].values, 
                                   val_edges_df['target_idx'].values], dtype=torch.long)
                                   
    # Test edges: Held-out set for final evaluation
    test_edges_df = edges_df[edges_df['source_year'] == TEST_YEAR]
    test_edge_index = torch.tensor([test_edges_df['source_idx'].values, 
                                    test_edges_df['target_idx'].values], dtype=torch.long)
    
    print("\n--- Data Loading Complete ---")
    print(f"Total Nodes: {num_nodes}")
    print(f"Training Edges (<= {TRAIN_YEAR_END}): {train_edge_index.size(1)}")
    print(f"Validation Edges ({VAL_YEAR}): {val_edge_index.size(1)}")
    print(f"Test Edges ({TEST_YEAR}): {test_edge_index.size(1)}")
    
    return x, train_edge_index, val_edge_index, test_edge_index, num_nodes

def train_epoch(encoder, predictor, x, train_edge_index, optimizer, criterion, device):
    encoder.train()
    predictor.train()
    optimizer.zero_grad()
    
    # 1. Get final node embeddings
    z = encoder(x, train_edge_index)
    
    # 2. Get positive training edges
    # For stability, we use the training edges as positive examples
    pos_edges = train_edge_index
    
    # 3. Sample negative edges
    neg_edges = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=z.size(0),
        num_neg_samples=pos_edges.size(1) # Match number of positive samples
    ).to(device)
    
    # Concatenate positive and negative edges
    all_edges = torch.cat([pos_edges, neg_edges], dim=1)
    
    # 4. Get predictions
    logits = predictor(z, all_edges).squeeze()
    
    # 5. Get labels
    pos_labels = torch.ones(pos_edges.size(1))
    neg_labels = torch.zeros(neg_edges.size(1))
    labels = torch.cat([pos_labels, neg_labels]).to(device)
    
    # 6. Calculate loss and backpropagate
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def test(encoder, predictor, x, train_edge_index, edges_to_test, device):
    encoder.eval()
    predictor.eval()
    
    # 1. Get final node embeddings (using the training graph structure)
    z = encoder(x, train_edge_index)
    
    # 2. Get positive edges and sample negative edges
    pos_edges = edges_to_test.to(device)
    neg_edges = negative_sampling(
        edge_index=train_edge_index, # Sample from the training graph
        num_nodes=z.size(0),
        num_neg_samples=pos_edges.size(1)
    ).to(device)
    
    all_edges = torch.cat([pos_edges, neg_edges], dim=1)
    
    # 3. Get predictions
    logits = predictor(z, all_edges).squeeze()
    
    # 4. Get labels
    pos_labels = torch.ones(pos_edges.size(1))
    neg_labels = torch.zeros(neg_edges.size(1))
    labels = torch.cat([pos_labels, neg_labels]).cpu()
    
    # 5. Calculate metrics
    preds = logits.cpu().sigmoid()
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    
    return auc, ap

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    x, train_edge_index, val_edge_index, test_edge_index, num_nodes = load_graph_data()
    
    # Move data to GPU
    x = x.to(device)
    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)
    
    # Initialize models
    encoder = GATEncoder(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, GAT_HEADS).to(device)
    predictor = LinkPredictor(OUT_CHANNELS, HIDDEN_CHANNELS).to(device)
    
    # Combine parameters for the optimizer
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("\n--- Starting Training ---")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(encoder, predictor, x, train_edge_index, optimizer, criterion, device)
        
        # Validation
        val_auc, val_ap = test(encoder, predictor, x, train_edge_index, val_edge_index, device)
        
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
        
    print("--- Training Complete ---")
    
    # Final Test
    print("Running final evaluation on Test Set...")
    test_auc, test_ap = test(encoder, predictor, x, train_edge_index, test_edge_index, device)
    print(f"Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
    # Save models
    print(f"Saving models to {ENCODER_MODEL_PATH} and {PREDICTOR_MODEL_PATH}")
    torch.save(encoder.state_dict(), ENCODER_MODEL_PATH)
    torch.save(predictor.state_dict(), PREDICTOR_MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
