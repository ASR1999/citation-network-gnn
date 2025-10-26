import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from tqdm import tqdm
import json
import os
import sys
import numpy as np # <-- IMPORT NUMPY

# Import our model classes
from model import GATEncoder, LinkPredictor

# --- Configuration ---
DATA_DIR = 'data'
NODES_FILE = os.path.join(DATA_DIR, 'nodes.csv')
EDGES_FILE = os.path.join(DATA_DIR, 'edges.csv')
NODE_FEATURES_FILE = os.path.join(DATA_DIR, 'node_features.pt')
NODE_MAP_FILE = os.path.join(DATA_DIR, 'paper_id_to_node_idx.json')

# Temporal split configuration
TRAIN_YEAR_END = 2017 # Train on all data up to and including 2017
VAL_YEAR = 2018      # Validate on data from this year
TEST_YEAR = 2019       # Test on data from this year

# Model Hyperparameters
IN_CHANNELS = 768 # Stays 768 for scibert
HIDDEN_CHANNELS = 256 
OUT_CHANNELS = 256
GAT_HEADS = 8
LEARNING_RATE = 0.001
EPOCHS = 10 

# Loader Hyperparameters
BATCH_SIZE = 2048 
NUM_NEIGHBORS = [30, 20] # 2-layer GAT
NUM_WORKERS = 8 

# Trained model output paths
ENCODER_MODEL_PATH = os.path.join(DATA_DIR, 'gat_encoder.pt')
PREDICTOR_MODEL_PATH = os.path.join(DATA_DIR, 'gat_predictor.pt')
# ---------------------

def load_graph_data():
    """
    Loads all necessary data from disk and performs the temporal split.
    """
    print("Loading graph data...")
    
    # 1. Load node features
    if not os.path.exists(NODE_FEATURES_FILE):
        print(f"Error: {NODE_FEATURES_FILE} not found. Please run '02_generate_features.py' first.")
        sys.exit()
    x = torch.load(NODE_FEATURES_FILE)
    num_nodes = x.size(0)
    
    # 2. Load node map
    if not os.path.exists(NODE_MAP_FILE):
        print(f"Error: {NODE_MAP_FILE} not found. Please run '02_generate_features.py' first.")
        sys.exit()
    with open(NODE_MAP_FILE, 'r') as f:
        paper_id_to_node_idx = json.load(f)
        
    # 3. Load node metadata (for year)
    # --- FIX: Specify dtype for paper_id ---
    nodes_df = pd.read_csv(NODES_FILE, dtype={'paper_id': str})
    # We need to map node_idx back to paper_id to get the year from nodes_df
    # --- FIX: Corrected typo from nodes_ to nodes_df ---
    if 'node_idx' not in nodes_df.columns:
        print("Warning: 'node_idx' not in nodes.csv, adding it from index.")
        nodes_df = nodes_df.reset_index().rename(columns={'index': 'node_idx'})
    
    node_idx_to_year = pd.Series(nodes_df.year.values, index=nodes_df.node_idx).to_dict()
    
    # 4. Load edges
    # --- FIX: Specify dtype for source_id and target_id ---
    print("Loading edges (this may take a moment)...")
    edges_df = pd.read_csv(EDGES_FILE, dtype={'source_id': str, 'target_id': str})
    
    # 5. Map string IDs to integer node indices
    print("Mapping edge IDs to integer indices...")
    edges_df['source_idx'] = edges_df['source_id'].map(paper_id_to_node_idx)
    edges_df['target_idx'] = edges_df['target_id'].map(paper_id_to_node_idx)
    
    # Drop edges that point to/from nodes not in our map
    original_edge_count = len(edges_df)
    edges_df = edges_df.dropna(subset=['source_idx', 'target_idx'])
    new_edge_count = len(edges_df)
    print(f"DEBUG: Dropped {original_edge_count - new_edge_count} edges that link to/from unknown nodes.")
    
    # Convert to integer indices
    edges_df['source_idx'] = edges_df['source_idx'].astype(int)
    edges_df['target_idx'] = edges_df['target_idx'].astype(int)
    
    # 6. Get year for each edge's SOURCE node
    edges_df['source_year'] = edges_df['source_idx'].map(node_idx_to_year)
    
    # --- FIX: Drop edges where the source node's year is unknown ---
    original_edge_count = len(edges_df)
    edges_df = edges_df.dropna(subset=['source_year'])
    new_edge_count = len(edges_df)
    print(f"DEBUG: Dropped {original_edge_count - new_edge_count} edges with missing source year.")
    
    # Now, safely convert year to integer
    edges_df['source_year'] = edges_df['source_year'].astype(int)
    # --- End Fix ---
    
    # 7. Perform temporal split
    print("Performing temporal split...")
    
    # --- FIX: Filter out 0-year edges BEFORE splitting ---
    valid_edges_df = edges_df[edges_df['source_year'] > 0]
    print(f"DEBUG: Found {len(valid_edges_df)} edges with a valid source year (> 0).")
    
    # Training edges: Used for message passing and training the predictor
    train_edges_df = valid_edges_df[valid_edges_df['source_year'] <= TRAIN_YEAR_END]
    
    # --- OPTIMIZATION: Stack with numpy first to avoid warning ---
    train_edge_index_np = np.stack([train_edges_df['source_idx'].values, 
                                    train_edges_df['target_idx'].values])
    train_edge_index = torch.tensor(train_edge_index_np, dtype=torch.long)
                                     
    # Validation edges: Used for validating the link predictor
    val_edges_df = valid_edges_df[valid_edges_df['source_year'] == VAL_YEAR]
    val_edge_index_np = np.stack([val_edges_df['source_idx'].values, 
                                  val_edges_df['target_idx'].values])
    val_edge_index = torch.tensor(val_edge_index_np, dtype=torch.long)
                                   
    # Test edges: Used for final hold-out evaluation
    test_edges_df = valid_edges_df[valid_edges_df['source_year'] == TEST_YEAR]
    test_edge_index_np = np.stack([test_edges_df['source_idx'].values, 
                                   test_edges_df['target_idx'].values])
    test_edge_index = torch.tensor(test_edge_index_np, dtype=torch.long)
    # --- END OPTIMIZATION ---
    
    print("\n--- Data Loading Complete ---")
    print(f"Total Nodes: {num_nodes}")
    print(f"Training Edges (<= {TRAIN_YEAR_END}): {train_edge_index.size(1)}")
    print(f"Validation Edges ({VAL_YEAR}): {val_edge_index.size(1)}")
    print(f"Test Edges ({TEST_YEAR}): {test_edge_index.size(1)}")
    
    if train_edge_index.size(1) == 0:
        print("\n❌ CRITICAL ERROR: Found 0 training edges after split.")
        print("Please check the 'year' distribution and your data files.")
        sys.exit()
    
    # Create the main Data object for the loaders
    # The loaders need the full graph topology (train_edge_index)
    # and the full set of node features (x).
    data = Data(x=x, edge_index=train_edge_index, num_nodes=num_nodes)
    
    return data, train_edge_index, val_edge_index, test_edge_index

def train_epoch(encoder, predictor, train_loader, optimizer, criterion, device):
    encoder.train()
    predictor.train()
    
    total_loss = 0
    total_batches = 0 # Keep track of the number of batches processed
    
    # Use tqdm for a progress bar over the mini-batches
    for batch in tqdm(train_loader, desc="Training Batches"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 1. Get final node embeddings for the sampled subgraph
        # batch.x contains the features for the nodes in the subgraph
        # batch.edge_index contains the topology of the subgraph
        try:
            z = encoder(batch.x, batch.edge_index)
            
            # 2. Get predictions for the positive and negative edges in the batch
            # batch.edge_label_index contains the edges to predict
            logits = predictor(z, batch.edge_label_index).squeeze()
            
            # 3. Get labels
            # batch.edge_label contains the 0/1 labels for the edges
            labels = batch.edge_label
            
            # 4. Calculate loss and backpropagate
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # --- FIX: Accumulate loss directly, don't multiply by num_graphs ---
            total_loss += loss.item() 
            total_batches += 1
            # --- END FIX ---

        except RuntimeError as e:
            print(f"\nCaught runtime error during training: {e}")
            print("This can sometimes happen with sparse data batches. Skipping batch.")
            optimizer.zero_grad() # Clear gradients even if skipping
            continue # Skip to the next batch
            
    # Return the average loss per batch for the epoch
    return total_loss / total_batches if total_batches > 0 else 0

@torch.no_grad()
def test(encoder, predictor, loader, device):
    encoder.eval()
    predictor.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Testing Batches"):
        batch = batch.to(device)
        
        try:
            # 1. Get final node embeddings
            z = encoder(batch.x, batch.edge_index)
            
            # 2. Get predictions
            logits = predictor(z, batch.edge_label_index).squeeze()
            
            # 3. Get labels
            labels = batch.edge_label
            
            all_preds.append(logits.cpu())
            all_labels.append(labels.cpu())
        except RuntimeError as e:
            print(f"\nCaught runtime error during testing: {e}")
            print("Skipping batch.")
            continue
    
    preds = torch.cat(all_preds, dim=0).sigmoid()
    labels = torch.cat(all_labels, dim=0)
    
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    
    return auc, ap

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data, train_edge_index, val_edge_index, test_edge_index = load_graph_data()
    
    # Put features on GPU.
    # Keep data (topology) on CPU for the loader.
    data.x = data.x.to(device)
    data = data.to('cpu') 
    
    print("\n--- Initializing DataLoaders ---")
    # This is the key change. We now use LinkNeighborLoader.
    # It samples neighbors and automatically provides negative edges.
    
    # Training loader
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        edge_label_index=train_edge_index,
        edge_label=torch.ones(train_edge_index.size(1)), # Positive edges
        neg_sampling_ratio=1.0, # 1 negative sample per positive sample
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True, # Helps speed up CPU-to-GPU data transfer
    )
    
    # Validation loader
    val_loader = LinkNeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        edge_label_index=val_edge_index,
        edge_label=torch.ones(val_edge_index.size(1)),
        neg_sampling_ratio=1.0,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )
    
    # Test loader
    test_loader = LinkNeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        edge_label_index=test_edge_index,
        edge_label=torch.ones(test_edge_index.size(1)),
        neg_sampling_ratio=1.0,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )
    
    # Initialize models with larger parameters
    encoder = GATEncoder(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, GAT_HEADS).to(device)
    predictor = LinkPredictor(OUT_CHANNELS, HIDDEN_CHANNELS).to(device)
    
    # Combine parameters for the optimizer
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("\n--- Starting Training ---")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(encoder, predictor, train_loader, optimizer, criterion, device)
        
        # Validation
        val_auc, val_ap = test(encoder, predictor, val_loader, device)
        
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
        
    print("--- Training Complete ---")
    
    # Final Test
    print("Running final evaluation on Test Set...")
    test_auc, test_ap = test(encoder, predictor, test_loader, device)
    print(f"Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
    # Save models
    print("Saving trained models...")
    torch.save(encoder.state_dict(), ENCODER_MODEL_PATH)
    torch.save(predictor.state_dict(), PREDICTOR_MODEL_PATH)
    print(f"Models saved to {DATA_DIR}")

if __name__ == "__main__":
    main()

