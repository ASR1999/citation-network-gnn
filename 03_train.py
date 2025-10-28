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
import numpy as np

# Import our model classes
from model import GATEncoder, LinkPredictor

# --- Configuration ---
DATA_DIR = 'data'
NODES_FILE = os.path.join(DATA_DIR, 'nodes.csv')
EDGES_FILE = os.path.join(DATA_DIR, 'edges.csv')
NODE_FEATURES_FILE = os.path.join(DATA_DIR, 'node_features.pt')
NODE_MAP_FILE = os.path.join(DATA_DIR, 'paper_id_to_node_idx.json')
# --- Directory to save epoch checkpoints ---
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Create checkpoint directory if it doesn't exist
# ---

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
EPOCHS = 10 # Max epochs

# Loader Hyperparameters
BATCH_SIZE = 2048
NUM_NEIGHBORS = [30, 20] # 2-layer GAT
NUM_WORKERS = 8

# Trained model output paths (These will now save the LAST epoch's model)
ENCODER_MODEL_PATH = os.path.join(DATA_DIR, 'gat_encoder_final.pt')
PREDICTOR_MODEL_PATH = os.path.join(DATA_DIR, 'gat_predictor_final.pt')

# --- Remove Best Model Paths ---
# BEST_ENCODER_MODEL_PATH = os.path.join(DATA_DIR, 'best_gat_encoder.pt')
# BEST_PREDICTOR_MODEL_PATH = os.path.join(DATA_DIR, 'best_gat_predictor.pt')
# ---

# --- Remove Early Stopping Config ---
# EARLY_STOPPING_PATIENCE = 3
# MIN_DELTA = 0.0001
# ---

def load_graph_data():
    """
    Loads all necessary data from disk and performs the temporal split.
    """
    # ... (load_graph_data function remains identical) ...
    print("Loading graph data...")

    # 1. Load node features
    if not os.path.exists(NODE_FEATURES_FILE):
        print(f"Error: {NODE_FEATURES_FILE} not found. Please run '02_generate_features.py' first.")
        sys.exit()
    print(f"Loading node features from {NODE_FEATURES_FILE}...")
    x = torch.load(NODE_FEATURES_FILE, weights_only=True) # Added weights_only
    num_nodes = x.size(0)

    # 2. Load node map
    if not os.path.exists(NODE_MAP_FILE):
        print(f"Error: {NODE_MAP_FILE} not found. Please run '02_generate_features.py' first.")
        sys.exit()
    print(f"Loading node map from {NODE_MAP_FILE}...")
    with open(NODE_MAP_FILE, 'r') as f:
        paper_id_to_node_idx = json.load(f)

    # 3. Load node metadata (for year)
    print(f"Loading node metadata from {NODES_FILE}...")
    nodes_df = pd.read_csv(NODES_FILE, dtype={'paper_id': str})
    if 'node_idx' not in nodes_df.columns:
        print("Warning: 'node_idx' not in nodes.csv, adding it from index.")
        nodes_df = nodes_df.reset_index().rename(columns={'index': 'node_idx'})

    node_idx_to_year = pd.Series(nodes_df.year.values, index=nodes_df.node_idx).to_dict()

    # 4. Load edges
    print(f"Loading edges from {EDGES_FILE} (this may take a moment)...")
    edges_df = pd.read_csv(EDGES_FILE, dtype={'source_id': str, 'target_id': str})

    # 5. Map string IDs to integer node indices
    print("Mapping edge IDs to integer indices...")
    edges_df['source_idx'] = edges_df['source_id'].map(paper_id_to_node_idx)
    edges_df['target_idx'] = edges_df['target_id'].map(paper_id_to_node_idx)

    original_edge_count = len(edges_df)
    edges_df = edges_df.dropna(subset=['source_idx', 'target_idx'])
    new_edge_count = len(edges_df)
    print(f"DEBUG: Dropped {original_edge_count - new_edge_count} edges that link to/from unknown nodes.")

    edges_df['source_idx'] = edges_df['source_idx'].astype(int)
    edges_df['target_idx'] = edges_df['target_idx'].astype(int)

    # 6. Get year for each edge's SOURCE node
    print("Mapping source years to edges...")
    edges_df['source_year'] = edges_df['source_idx'].map(node_idx_to_year)

    original_edge_count = len(edges_df)
    edges_df = edges_df.dropna(subset=['source_year'])
    new_edge_count = len(edges_df)
    print(f"DEBUG: Dropped {original_edge_count - new_edge_count} edges with missing source year.")

    edges_df['source_year'] = edges_df['source_year'].astype(int)

    # 7. Perform temporal split
    print("Performing temporal split...")
    valid_edges_df = edges_df[edges_df['source_year'] > 0]
    print(f"DEBUG: Found {len(valid_edges_df)} edges with a valid source year (> 0).")

    train_edges_df = valid_edges_df[valid_edges_df['source_year'] <= TRAIN_YEAR_END]
    train_edge_index_np = np.stack([train_edges_df['source_idx'].values,
                                    train_edges_df['target_idx'].values])
    train_edge_index = torch.tensor(train_edge_index_np, dtype=torch.long)

    val_edges_df = valid_edges_df[valid_edges_df['source_year'] == VAL_YEAR]
    val_edge_index_np = np.stack([val_edges_df['source_idx'].values,
                                  val_edges_df['target_idx'].values])
    val_edge_index = torch.tensor(val_edge_index_np, dtype=torch.long)

    test_edges_df = valid_edges_df[valid_edges_df['source_year'] == TEST_YEAR]
    test_edge_index_np = np.stack([test_edges_df['source_idx'].values,
                                   test_edges_df['target_idx'].values])
    test_edge_index = torch.tensor(test_edge_index_np, dtype=torch.long)

    print("\n--- Data Loading Complete ---")
    print(f"Total Nodes: {num_nodes}")
    print(f"Training Edges (<= {TRAIN_YEAR_END}): {train_edge_index.size(1)}")
    print(f"Validation Edges ({VAL_YEAR}): {val_edge_index.size(1)}")
    print(f"Test Edges ({TEST_YEAR}): {test_edge_index.size(1)}")

    if train_edge_index.size(1) == 0:
        print("\n❌ CRITICAL ERROR: Found 0 training edges after split.")
        sys.exit()

    data = Data(x=x, edge_index=train_edge_index, num_nodes=num_nodes)

    return data, train_edge_index, val_edge_index, test_edge_index


def train_epoch(encoder, predictor, train_loader, optimizer, criterion, device):
    # ... (train_epoch function remains identical) ...
    encoder.train()
    predictor.train()

    total_loss = 0
    total_batches = 0

    for batch in tqdm(train_loader, desc="Training Batches"):
        batch = batch.to(device)
        optimizer.zero_grad()

        try:
            z = encoder(batch.x, batch.edge_index)
            logits = predictor(z, batch.edge_label_index).squeeze()
            labels = batch.edge_label
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                 print("\nWARNING: CUDA out of memory. Skipping batch.")
                 optimizer.zero_grad() # Clear gradients before skipping
                 torch.cuda.empty_cache() # Try to free memory
                 continue # Skip to next batch
            else:
                 print(f"\nCaught runtime error during training: {e}")
                 print("Skipping batch.")
                 optimizer.zero_grad()
                 continue

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
            z = encoder(batch.x, batch.edge_index)
            logits = predictor(z, batch.edge_label_index).squeeze()
            labels = batch.edge_label
            all_preds.append(logits.cpu())
            all_labels.append(labels.cpu())
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                 print("\nWARNING: CUDA out of memory during testing. Skipping batch.")
                 torch.cuda.empty_cache()
                 continue
            else:
                 print(f"\nCaught runtime error during testing: {e}")
                 print("Skipping batch.")
                 continue

    if not all_preds: # Handle case where all batches failed
        return 0.0, 0.0

    preds = torch.cat(all_preds, dim=0).sigmoid()
    labels = torch.cat(all_labels, dim=0)

    # Handle potential NaN or inf values if predictions are bad
    if not torch.isfinite(preds).all():
        print("\nWARNING: Non-finite values (NaN or inf) found in predictions during testing. AUC/AP might be unreliable.")
        return 0.0, 0.0 # Return default values

    # Ensure labels are binary (0 or 1)
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()

    # Check if labels contain both classes for AUC calculation
    if len(np.unique(labels)) < 2:
        print("\nWARNING: Only one class present in labels during testing. AUC is not defined.")
        auc = 0.0 # Or handle as appropriate
    else:
        try:
           auc = roc_auc_score(labels, preds)
        except ValueError as e:
           print(f"\nError calculating AUC: {e}")
           auc = 0.0

    try:
        ap = average_precision_score(labels, preds)
    except ValueError as e:
        print(f"\nError calculating AP: {e}")
        ap = 0.0


    return auc, ap

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data, train_edge_index, val_edge_index, test_edge_index = load_graph_data()

    # Move features to GPU only once, keep topology on CPU for loader
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to('cpu') # Ensure edge_index is on CPU

    print("\n--- Initializing DataLoaders ---")
    data_for_loader = Data(
        x=data.x.cpu(), # Loader expects features on CPU initially
        edge_index=data.edge_index,
        num_nodes=data.num_nodes
    )

    train_loader = LinkNeighborLoader(
        data_for_loader, num_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE,
        edge_label_index=train_edge_index, edge_label=torch.ones(train_edge_index.size(1)),
        neg_sampling_ratio=1.0, num_workers=NUM_WORKERS, shuffle=True,
        pin_memory=True if device == 'cuda' else False,
    )
    val_loader = LinkNeighborLoader(
        data_for_loader, num_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE,
        edge_label_index=val_edge_index, edge_label=torch.ones(val_edge_index.size(1)),
        neg_sampling_ratio=1.0, num_workers=NUM_WORKERS, shuffle=False,
        pin_memory=True if device == 'cuda' else False,
    )
    test_loader = LinkNeighborLoader(
        data_for_loader, num_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE,
        edge_label_index=test_edge_index, edge_label=torch.ones(test_edge_index.size(1)),
        neg_sampling_ratio=1.0, num_workers=NUM_WORKERS, shuffle=False,
        pin_memory=True if device == 'cuda' else False,
    )

    encoder = GATEncoder(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, GAT_HEADS).to(device)
    predictor = LinkPredictor(OUT_CHANNELS, HIDDEN_CHANNELS, 1).to(device)

    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("\n--- Starting Training ---")
    last_epoch = 0 # Keep track of the last epoch completed
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(encoder, predictor, train_loader, optimizer, criterion, device)
        val_auc, val_ap = test(encoder, predictor, val_loader, device)

        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")

        # --- Save model weights after EVERY epoch ---
        epoch_encoder_path = os.path.join(CHECKPOINT_DIR, f'gat_encoder_epoch_{epoch:02d}.pt')
        epoch_predictor_path = os.path.join(CHECKPOINT_DIR, f'gat_predictor_epoch_{epoch:02d}.pt')
        print(f"  Saving model checkpoint for epoch {epoch} to {CHECKPOINT_DIR}")
        torch.save(encoder.state_dict(), epoch_encoder_path)
        torch.save(predictor.state_dict(), epoch_predictor_path)
        last_epoch = epoch # Update last completed epoch

    print("--- Training Complete ---")

    # Final Test using the model from the LAST completed epoch
    print(f"Loading model weights from last epoch ({last_epoch}) for final test evaluation...")
    try:
        # Construct paths for the last epoch's saved model
        final_encoder_path = os.path.join(CHECKPOINT_DIR, f'gat_encoder_epoch_{last_epoch:02d}.pt')
        final_predictor_path = os.path.join(CHECKPOINT_DIR, f'gat_predictor_epoch_{last_epoch:02d}.pt')

        encoder.load_state_dict(torch.load(final_encoder_path))
        predictor.load_state_dict(torch.load(final_predictor_path))
        print(f"Model from epoch {last_epoch} loaded successfully.")
    except FileNotFoundError:
        print("ERROR: Model files for the last epoch not found. Cannot perform final test.")
        sys.exit() # Exit if no weights can be loaded
    except Exception as e: # Catch other potential loading errors
        print(f"ERROR: Could not load model weights for epoch {last_epoch}: {e}")
        sys.exit()


    print("Running final evaluation on Test Set...")
    test_auc, test_ap = test(encoder, predictor, test_loader, device)
    print(f"Final Test AUC (using epoch {last_epoch} model): {test_auc:.4f}, Final Test AP: {test_ap:.4f}")

    # --- Save the final model weights (optional, but good practice, points to last epoch) ---
    torch.save(encoder.state_dict(), ENCODER_MODEL_PATH)
    torch.save(predictor.state_dict(), PREDICTOR_MODEL_PATH)
    print(f"Final models (from epoch {last_epoch}) also saved as '{os.path.basename(ENCODER_MODEL_PATH)}' and '{os.path.basename(PREDICTOR_MODEL_PATH)}'")
    # ---

if __name__ == "__main__":
    main()