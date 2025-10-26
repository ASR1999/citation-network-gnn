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

# Use a smaller, efficient model if SciBERT is too slow/large
MODEL_NAME = 'allenai/scibert_scivocab_uncased' # 768-dim
# MODEL_NAME = 'distilbert-base-uncased' # 768-dim, but faster
# MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' # 384-dim, very fast
# BATCH_SIZE = 64 # Adjust based on your GPU VRAM
BATCH_SIZE = 1024
# ---------------------

def generate_features():
    """
    Loads the preprocessed nodes.csv, generates node features using a
    pre-trained model (like SciBERT or DistilBERT), and saves:
    1. node_features.pt: A [num_nodes, feature_dim] tensor.
    2. paper_id_to_node_idx.json: A mapping from string paper_id to integer node index.
    """
    
    print(f"Loading preprocessed nodes from {NODES_FILE}...")
    if not os.path.exists(NODES_FILE):
        print(f"Error: {NODES_FILE} not found. Please run '01_preprocess_data.py' first.")
        return
        
    nodes_df = pd.read_csv(NODES_FILE).fillna('')
    
    # 1. Create and save the paper_id to node_idx mapping
    print("Creating paper_id to node_idx mapping...")
    # Ensure a consistent integer index
    nodes_df = nodes_df.reset_index().rename(columns={'index': 'node_idx'})
    paper_id_to_node_idx = pd.Series(nodes_df.node_idx.values, index=nodes_df.paper_id).to_dict()
    
    with open(NODE_MAP_FILE, 'w') as f:
        json.dump(paper_id_to_node_idx, f)
    print(f"Saved mapping to {NODE_MAP_FILE}")
    
    # 2. Set up model and tokenizer
    print(f"Loading pre-trained model: {MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    # 3. Generate embeddings in batches
    print(f"Generating node embeddings in batches of {BATCH_SIZE}...")
    all_texts = (nodes_df['title'] + " " + nodes_df['abstract']).tolist()
    all_embeddings = []
    
    for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Generating Embeddings"):
        batch_texts = all_texts[i:i + BATCH_SIZE]
        
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the embedding of the [CLS] token
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        all_embeddings.append(cls_embeddings)
        
    # Concatenate all batch embeddings into a single tensor
    x = torch.cat(all_embeddings, dim=0)
    
    # 4. Save the final tensor
    torch.save(x, NODE_FEATURES_FILE)
    
    print("\n--- Feature Generation Complete ---")
    print(f"Node features tensor shape: {x.shape}")
    print(f"Saved node features to {NODE_FEATURES_FILE}")

if __name__ == "__main__":
    generate_features()
