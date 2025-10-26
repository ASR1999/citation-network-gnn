import os
import pandas as pd
import torch
import json
from tqdm import tqdm

print("--- Starting Data Health Check ---")

DATA_DIR = 'data'
FILES_TO_CHECK = {
    "nodes.csv": os.path.join(DATA_DIR, 'nodes.csv'),
    "edges.csv": os.path.join(DATA_DIR, 'edges.csv'),
    "node_features.pt": os.path.join(DATA_DIR, 'node_features.pt'),
    "node_map.json": os.path.join(DATA_DIR, 'paper_id_to_node_idx.json')
}

# 1. Check File Existence
print("\n[Check 1: File Existence]")
all_files_exist = True
for name, path in FILES_TO_CHECK.items():
    if not os.path.exists(path):
        print(f"❌ ERROR: File not found: {path}")
        all_files_exist = False
    else:
        print(f"✅ Found: {path}")

if not all_files_exist:
    print("\n--- ABORTING: Not all required files were found. ---")
    print("Please run '01_preprocess_data.py' and '02_generate_features.py' first.")
    exit()

# 2. Check nodes.csv and year distribution
print("\n[Check 2: nodes.csv Analysis]")
try:
    # --- FIX: Specify dtype for paper_id ---
    nodes_df = pd.read_csv(FILES_TO_CHECK["nodes.csv"], dtype={'paper_id': str})
    print(f"✅ Loaded nodes.csv. Shape: {nodes_df.shape}")
    
    if 'year' not in nodes_df.columns:
        print("❌ ERROR: 'year' column not in nodes.csv")
        exit()
        
    print("Value counts for 'year' column (Top 15):")
    # Fill NaN before value_counts to see them
    print(nodes_df['year'].fillna('NaN').value_counts().head(15).to_string())
    
    # Describe on non-NaN years
    year_stats = nodes_df['year'].dropna().describe()
    print("\nStatistics for 'year' column (non-NaN):")
    print(year_stats.to_string())
    
    if year_stats['min'] < 1900 or year_stats['max'] > 2025:
        print(f"⚠️ WARNING: 'year' range seems suspicious (min: {year_stats['min']}, max: {year_stats['max']}).")
        
    nan_years = nodes_df['year'].isna().sum()
    zero_years = (nodes_df['year'] == 0).sum()
    print(f"Nodes with NaN year: {nan_years}")
    print(f"Nodes with '0' year: {zero_years}")

except Exception as e:
    print(f"❌ ERROR loading or analyzing nodes.csv: {e}")
    exit()

# 3. Check node_features.pt
print("\n[Check 3: node_features.pt Analysis]")
try:
    x = torch.load(FILES_TO_CHECK["node_features.pt"])
    print(f"✅ Loaded node_features.pt. Shape: {x.shape}")
    
    if x.shape[0] != nodes_df.shape[0]:
        print(f"❌ ERROR: Node feature count ({x.shape[0]}) does not match nodes.csv count ({nodes_df.shape[0]})!")
    else:
        print("✅ Node feature count matches nodes.csv.")
        
    if x.shape[1] != 768:
        print(f"⚠️ WARNING: Node feature dimension is {x.shape[1]} (expected 768 for scibert).")
    
except Exception as e:
    print(f"❌ ERROR loading or analyzing node_features.pt: {e}")
    exit()

# 4. Check edges.csv and map integrity
print("\n[Check 4: edges.csv and Mapping Analysis]")
try:
    # --- FIX: Specify dtype for source_id and target_id ---
    edges_df = pd.read_csv(FILES_TO_CHECK["edges.csv"], dtype={'source_id': str, 'target_id': str})
    print(f"✅ Loaded edges.csv. Shape: {edges_df.shape}")

    with open(FILES_TO_CHECK["node_map.json"], 'r') as f:
        node_map = json.load(f)
    print(f"✅ Loaded node_map.json. Total mapped nodes: {len(node_map)}")

    # Check mapping
    print("Checking edge-to-node mapping (this may take a moment)...")
    source_in_map = edges_df['source_id'].isin(node_map.keys()).sum()
    target_in_map = edges_df['target_id'].isin(node_map.keys()).sum()
    
    print(f"✅ {source_in_map / len(edges_df) * 100:.2f}% of 'source_id's are in the node map.")
    print(f"✅ {target_in_map / len(edges_df) * 100:.2f}% of 'target_id's are in the node map.")

    if source_in_map < len(edges_df) or target_in_map < len(edges_df):
        print("⚠️ INFO: Some edges point to nodes not in our dataset. They will be dropped (this is normal).")

except Exception as e:
    print(f"❌ ERROR loading or analyzing edges.csv / node_map.json: {e}")
    exit()

# 5. Dry-Run of Temporal Split
print("\n[Check 5: Temporal Split Dry-Run]")
try:
    # Re-run the logic from 03_train.py
    # --- FIX: Ensure nodes_df is read correctly here too ---
    nodes_df = pd.read_csv(FILES_TO_CHECK["nodes.csv"], dtype={'paper_id': str})
    if 'node_idx' not in nodes_df.columns:
         nodes_df = nodes_df.reset_index().rename(columns={'index': 'node_idx'})
    node_idx_to_year = pd.Series(nodes_df.year.values, index=nodes_df.node_idx).to_dict()
    
    # --- FIX: Ensure edges_df is read correctly here too ---
    edges_df = pd.read_csv(FILES_TO_CHECK["edges.csv"], dtype={'source_id': str, 'target_id': str})
    edges_df['source_idx'] = edges_df['source_id'].map(node_map)
    edges_df['target_idx'] = edges_df['target_id'].map(node_map)
    edges_df = edges_df.dropna(subset=['source_idx', 'target_idx'])
    
    edges_df['source_year'] = edges_df['source_idx'].map(node_idx_to_year)
    edges_df = edges_df.dropna(subset=['source_year'])
    edges_df['source_year'] = edges_df['source_year'].astype(int)

    # --- This is the key logic ---
    TRAIN_YEAR_END = 2017
    VAL_YEAR = 2018
    TEST_YEAR = 2019
    
    # We must filter out '0' years, as these are invalid for splitting
    valid_edges_df = edges_df[edges_df['source_year'] > 0]
    
    train_edges_df = valid_edges_df[valid_edges_df['source_year'] <= TRAIN_YEAR_END]
    val_edges_df = valid_edges_df[valid_edges_df['source_year'] == VAL_YEAR]
    test_edges_df = valid_edges_df[valid_edges_df['source_year'] == TEST_YEAR]
    
    print("--- Dry-Run Results ---")
    print(f"Total valid edges (year > 0): {len(valid_edges_df)}")
    print(f"Training Edges (<= {TRAIN_YEAR_END}): {len(train_edges_df)}")
    print(f"Validation Edges ({VAL_YEAR}): {len(val_edges_df)}")
    print(f"Test Edges ({TEST_YEAR}): {len(test_edges_df)}")

    if len(train_edges_df) == 0:
        print("\n❌ CRITICAL ERROR: Found 0 training edges.")
        print("This means no edges in your 'edges.csv' have a source paper with a year <= 2017.")
        print("Please check the 'year' distribution from Check 2.")
    else:
        print("\n✅ Temporal split seems successful.")

except Exception as e:
    print(f"❌ ERROR during temporal split dry-run: {e}")
    exit()

print("\n--- Data Health Check Complete ---")