import torch
import torch.nn.functional as F
import pandas as pd
import json
import os
from transformers import AutoTokenizer, AutoModel

# Import our model class
from model import GATEncoder

# --- Configuration ---
DATA_DIR = 'data'
NODES_FILE = os.path.join(DATA_DIR, 'nodes.csv')
EDGES_FILE = os.path.join(DATA_DIR, 'edges.csv') # Needed for train_edge_index
NODE_FEATURES_FILE = os.path.join(DATA_DIR, 'node_features.pt')
NODE_MAP_FILE = os.path.join(DATA_DIR, 'paper_id_to_node_idx.json')
ENCODER_MODEL_PATH = os.path.join(DATA_DIR, 'gat_encoder.pt')

# Must match the models used in training
FEATURE_MODEL_NAME = 'distilbert-base-uncased' # 768-dim
IN_CHANNELS = 768
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 128
GAT_HEADS = 4
TRAIN_YEAR_END = 2017 # Must match 03_train.py
# ---------------------

class Recommender:
    def __init__(self):
        print("Initializing recommender...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load SciBERT (or other) model for embedding queries
        print(f"Loading feature model: {FEATURE_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(FEATURE_MODEL_NAME)
        self.feature_model = AutoModel.from_pretrained(FEATURE_MODEL_NAME).to(self.device)
        self.feature_model.eval()
        
        # 2. Load the trained GAT encoder
        print("Loading trained GAT encoder...")
        self.encoder = GATEncoder(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, GAT_HEADS).to(self.device)
        self.encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=self.device))
        self.encoder.eval()
        
        # 3. Load node metadata
        print("Loading node metadata...")
        self.nodes_df = pd.read_csv(NODES_FILE)
        
        # 4. Load all initial features
        print("Loading initial node features...")
        self.x_all = torch.load(NODE_FEATURES_FILE).to(self.device)
        
        # 5. Load and build the training graph (needed for the encoder)
        print("Loading and building training graph...")
        with open(NODE_MAP_FILE, 'r') as f:
            paper_id_to_node_idx = json.load(f)
        
        node_idx_to_year = pd.Series(self.nodes_df.year.values, index=self.nodes_df.node_idx).to_dict()
        edges_df = pd.read_csv(EDGES_FILE)
        edges_df['source_idx'] = edges_df['source_id'].map(paper_id_to_node_idx)
        edges_df['target_idx'] = edges_df['target_id'].map(paper_id_to_node_idx)
        edges_df = edges_df.dropna(subset=['source_idx', 'target_idx'])
        edges_df['source_idx'] = edges_df['source_idx'].astype(int)
        edges_df['target_idx'] = edges_df['target_idx'].astype(int)
        edges_df['source_year'] = edges_df['source_idx'].map(node_idx_to_year)
        
        train_edges_df = edges_df[edges_df['source_year'] <= TRAIN_YEAR_END]
        self.train_edge_index = torch.tensor([train_edges_df['source_idx'].values, 
                                              train_edges_df['target_idx'].values], dtype=torch.long).to(self.device)
                                     
        # 6. Generate final, graph-aware embeddings FOR ALL PAPERS
        # This is a one-time calculation
        print("Generating final graph-aware embeddings for all papers...")
        with torch.no_grad():
            self.z_all = self.encoder(self.x_all, self.train_edge_index).cpu()
            
        print("\n--- Recommender Ready ---")

    def _get_query_embedding(self, text):
        """Helper to get the initial SciBERT/DistilBERT embedding for a query text."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.feature_model(**inputs)
        
        # Use [CLS] token embedding
        return outputs.last_hidden_state[:, 0, :].cpu()

    def recommend(self, query_title, query_abstract, k=10):
        """
        Recommends the top-k papers to cite for a new query paper.
        """
        # 1. Get the initial embedding for the query paper
        query_text = query_title + " " + query_abstract
        x_query = self._get_query_embedding(query_text)
        
        # 2. Compute similarity
        # We compare the query's *initial* embedding (x_query) against
        # the *graph-aware* embeddings of all papers in the database (z_all).
        # This compares the query's "content" to the database's "content-in-context".
        sim = F.cosine_similarity(x_query, self.z_all)
        
        # 3. Get top-k recommendations
        top_k_scores, top_k_indices = torch.topk(sim, k)
        
        # 4. Look up metadata
        results = self.nodes_df.iloc[top_k_indices.numpy()].copy()
        results['similarity_score'] = top_k_scores.numpy()
        
        return results[['node_idx', 'title', 'year', 'similarity_score']]

if __name__ == "__main__":
    recommender = Recommender()

    # --- DEMO ---
    # This is the abstract from the original "GAT" paper
    demo_title = "Graph Attention Networks"
    demo_abstract = ("We present graph attention networks (GATs), novel neural network "
                     "architectures that operate on graph-structured data, leveraging "
                     "masked self-attentional layers to address the shortcomings of "
                     "prior methods based on graph convolutions or their approximations. "
                     "By stacking layers in which nodes are able to attend over their "
                     "neighborhoods' features, we enable (implicitly) specifying "
                     "different weights to different nodes in a neighborhood, without "
                     "requiring any kind of costly matrix operation (such as inversion) "
                     "or depending on knowing the graph structure upfront.")

    print(f"\n--- Recommending citations for: '{demo_title}' ---")
    recommendations = recommender.recommend(demo_title, demo_abstract, k=10)
    
    print(recommendations.to_string())
