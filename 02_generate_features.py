import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import json

# --- Configuration ---
NODES_FILE = 'data/nodes.csv'
OUTPUT_DIR = 'data'
NODE_FEATURES_FILE = os.path.join(OUTPUT_DIR, 'node_features.pt')
NODE_MAP_FILE = os.path.join(OUTPUT_DIR, 'paper_id_to_node_idx.json')

# We have an A100, let's use the full SciBERT model.
MODEL_NAME = 'allenai/scibert_scivocab_uncased' # 768-dim
# An A100 can handle a large batch size.
BATCH_SIZE = 512 
# ---------------------

def generate_features():
    if not os.path.exists(NODES_FILE):
        print(f"Error: {NODES_FILE} not found. Please run '01_preprocess_data.py' first.")
        return

    print("--- Starting Feature Generation ---")
    
    # 1. Load nodes
    # --- FIX: Specify dtype for paper_id ---
    print(f"Loading {NODES_FILE}...")
    nodes_df = pd.read_csv(NODES_FILE, dtype={'paper_id': str}).fillna('')
    print(f"Loaded {len(nodes_df)} nodes.")

    # 2. Create paper_id to sequential node_idx mapping
    # This is crucial. The graph needs 0-indexed integer IDs.
    print("Creating paper_id to node_idx map...")
    nodes_df = nodes_df.reset_index().rename(columns={'index': 'node_idx'})
    # --- FIX: Ensure paper_id is string type for map keys ---
    paper_id_to_idx = pd.Series(nodes_df.node_idx.values, index=nodes_df.paper_id.astype(str)).to_dict()
    
    with open(NODE_MAP_FILE, 'w') as f:
        json.dump(paper_id_to_idx, f)
    print(f"Saved map to {NODE_MAP_FILE}")

    # 3. Generate embeddings
    print(f"Loading pre-trained model: {MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    print(f"Generating node embeddings in batches of {BATCH_SIZE}...")
    all_texts = (nodes_df['title'] + " " + nodes_df['abstract']).tolist()
    all_embeddings = []
    
    for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Generating Embeddings"):
        batch_texts = all_texts[i:i + BATCH_SIZE]
        
        # Tokenize batch
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the embedding of the [CLS] token
        # [batch_size, sequence_length, hidden_size] -> [batch_size, hidden_size]
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        all_embeddings.append(cls_embeddings)
        
    # Combine all batch embeddings
    x = torch.cat(all_embeddings, dim=0)
    
    # 4. Save the final tensor
    print(f"\n--- Feature Generation Complete ---")
    print(f"Node features tensor shape: {x.shape}")
    torch.save(x, NODE_FEATURES_FILE)
    print(f"Saved node features to {NODE_FEATURES_FILE}")

if __name__ == "__main__":
    generate_features()

# --- Feature Generation Complete ---
# Node features tensor shape: torch.Size([4894402, 768])
# Saved node features to data/node_features.pt