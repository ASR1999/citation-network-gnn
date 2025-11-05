import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader # Import NeighborLoader
from torch_geometric.data import Data # Needed for NeighborLoader input
import pandas as pd
import json
import os
from tqdm import tqdm # <--- ADD THIS IMPORT

# Try importing safetensors, handle if not installed
try:
    import safetensors
    SAFE_TENSORS_AVAILABLE = True
except ImportError:
    SAFE_TENSORS_AVAILABLE = False
    print("Warning: 'safetensors' library not found. Falling back to default PyTorch loading.")
    print("Install with: pip install safetensors")

from transformers import AutoTokenizer, AutoModel
from model import GATEncoder
import sys
import numpy as np # Needed for numpy arrays in tensor creation

# Import our model class
# from model import GATEncoder # Make sure model.py is in the same directory

# --- Configuration ---
DATA_DIR = 'data'
NODES_FILE = os.path.join(DATA_DIR, 'nodes.csv')
EDGES_FILE = os.path.join(DATA_DIR, 'edges.csv')
NODE_FEATURES_FILE = os.path.join(DATA_DIR, 'node_features.pt')
NODE_MAP_FILE = os.path.join(DATA_DIR, 'paper_id_to_node_idx.json')

# --- Select which epoch's checkpoint to load ---
LOAD_EPOCH = 2 # Change this based on your training results
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')
ENCODER_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'gat_encoder_epoch_{LOAD_EPOCH:02d}.pt')

# --- MUST match the models used in TRAINING (03_train.py) ---
FEATURE_MODEL_NAME = 'allenai/scibert_scivocab_uncased'
IN_CHANNELS = 768
HIDDEN_CHANNELS = 256
OUT_CHANNELS = 256
GAT_HEADS = 8
TRAIN_YEAR_END = 2017
# --- Inference Batching Config ---
INFERENCE_BATCH_SIZE = 1024 # Adjust based on VRAM (larger is faster if it fits)
INFERENCE_NUM_NEIGHBORS = [-1] * len([30, 20]) # Use all neighbors during inference (same depth as training)
INFERENCE_NUM_WORKERS = 4 # Adjust based on CPU cores
# ---------------------

class Recommender:
    def __init__(self):
        print("Initializing recommender...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Load Feature model (SciBERT)
        print(f"Loading feature model: {FEATURE_MODEL_NAME}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(FEATURE_MODEL_NAME)
            load_args = {'use_safetensors': True} if SAFE_TENSORS_AVAILABLE else {}
            print(f"Attempting to load feature model with args: {load_args}")
            self.feature_model = AutoModel.from_pretrained(FEATURE_MODEL_NAME, **load_args).to(self.device)
            self.feature_model.eval()
        except RuntimeError as e:
            if "upgrade torch to at least v2.6" in str(e):
                 print("\nFATAL ERROR: Your PyTorch version is too old and insecure for torch.load.")
                 print("The 'safetensors' workaround did not prevent the error.")
                 print("Consider creating a new environment with Python 3.11/3.12 and the latest PyTorch/PyG.")
            else:
                 print(f"\nRuntime error loading feature model: {e}")
            sys.exit()
        except Exception as e:
            print(f"Error loading feature model: {e}")
            sys.exit()

        # 2. Load the trained GAT encoder
        print(f"Loading trained GAT encoder from epoch {LOAD_EPOCH}...")
        if not os.path.exists(ENCODER_MODEL_PATH):
             print(f"Error: Encoder model file not found at {ENCODER_MODEL_PATH}")
             sys.exit()
        try:
            self.encoder = GATEncoder(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, GAT_HEADS).to(self.device)
            self.encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=self.device, weights_only=True))
            self.encoder.eval()
        except RuntimeError as e:
            print(f"Error loading encoder state_dict (likely hyperparameter mismatch): {e}")
            sys.exit()
        except Exception as e:
            print(f"Error loading encoder model: {e}")
            sys.exit()


        # 3. Load node metadata and map
        print("Loading node metadata and map...")
        try:
            self.nodes_df = pd.read_csv(NODES_FILE, dtype={'paper_id': str})
            if 'node_idx' not in self.nodes_df.columns:
                 self.nodes_df = self.nodes_df.reset_index().rename(columns={'index': 'node_idx'})
            with open(NODE_MAP_FILE, 'r') as f:
                self.paper_id_to_node_idx = json.load(f)
            self.node_idx_to_metadata = self.nodes_df.set_index('node_idx').to_dict('index')
        except FileNotFoundError as e:
            print(f"Error: Required data file not found: {e}")
            sys.exit()
        except Exception as e:
            print(f"Error loading metadata/map: {e}")
            sys.exit()


        # 4. Load all initial features (Keep on CPU initially)
        print(f"Loading initial node features from {NODE_FEATURES_FILE}...")
        try:
             self.x_all = torch.load(NODE_FEATURES_FILE, map_location='cpu', weights_only=True)
             self.num_nodes = self.x_all.size(0)
             print(f"Initial features shape: {self.x_all.shape}")
        except FileNotFoundError:
            print(f"Error: Node features file not found at {NODE_FEATURES_FILE}")
            sys.exit()
        except RuntimeError as e:
            if "upgrade torch to at least v2.6" in str(e):
                 print("\nFATAL ERROR: Your PyTorch version is too old and insecure for torch.load (when loading features).")
            else:
                 print(f"\nRuntime error loading node features: {e}")
            sys.exit()
        except Exception as e:
            print(f"Error loading node features: {e}")
            sys.exit()

        # 5. Load and build the training graph structure (on CPU)
        print("Loading and building training graph structure (on CPU)...")
        try:
            node_idx_to_year = pd.Series(self.nodes_df.year.values, index=self.nodes_df.node_idx).to_dict()
            edges_df = pd.read_csv(EDGES_FILE, dtype={'source_id': str, 'target_id': str})
            edges_df['source_idx'] = edges_df['source_id'].map(self.paper_id_to_node_idx)
            edges_df['target_idx'] = edges_df['target_id'].map(self.paper_id_to_node_idx)
            edges_df = edges_df.dropna(subset=['source_idx', 'target_idx'])
            edges_df['source_idx'] = edges_df['source_idx'].astype(int)
            edges_df['target_idx'] = edges_df['target_idx'].astype(int)
            edges_df['source_year'] = edges_df['source_idx'].apply(lambda idx: node_idx_to_year.get(idx, 0))
            edges_df = edges_df.dropna(subset=['source_year'])
            edges_df['source_year'] = edges_df['source_year'].astype(int)
            train_edges_df = edges_df[(edges_df['source_year'] > 0) & (edges_df['source_year'] <= TRAIN_YEAR_END)]
            train_edge_index_np = np.stack([train_edges_df['source_idx'].values,
                                            train_edges_df['target_idx'].values])
            self.train_edge_index = torch.tensor(train_edge_index_np, dtype=torch.long)
            print(f"Training graph edge_index shape: {self.train_edge_index.shape}")
        except FileNotFoundError as e:
             print(f"Error: Edges file not found: {e}")
             sys.exit()
        except Exception as e:
             print(f"Error processing edges: {e}")
             sys.exit()


        # 6. Generate final, graph-aware embeddings FOR ALL PAPERS using BATCHED INFERENCE
        print("Generating final graph-aware embeddings for all papers (using batched inference)...")
        self.z_all = torch.zeros((self.num_nodes, OUT_CHANNELS), device='cpu') # Store final embeddings on CPU

        # Create Data object for NeighborLoader (features on CPU)
        inference_data = Data(x=self.x_all, edge_index=self.train_edge_index, num_nodes=self.num_nodes)

        # Create NeighborLoader to iterate through all nodes
        inference_loader = NeighborLoader(
             inference_data,
             num_neighbors=INFERENCE_NUM_NEIGHBORS,
             batch_size=INFERENCE_BATCH_SIZE,
             input_nodes=torch.arange(self.num_nodes), # Process all nodes
             shuffle=False, # Order matters for reconstruction
             num_workers=INFERENCE_NUM_WORKERS,
             pin_memory=True if self.device == 'cuda' else False,
        )

        try:
            with torch.no_grad():
                # *** Use the imported tqdm here ***
                for i, batch in enumerate(tqdm(inference_loader, desc="Generating Embeddings Batches")):
                    batch = batch.to(self.device)
                    # Run encoder only on the sampled subgraph for the current batch
                    batch_z = self.encoder(batch.x, batch.edge_index)
                    # Get embeddings ONLY for the target nodes of this batch
                    target_node_embeddings = batch_z[:batch.batch_size]
                    # Store these embeddings in the correct positions in the full z_all tensor (on CPU)
                    self.z_all[batch.input_id[:batch.batch_size]] = target_node_embeddings.cpu()

            print(f"Final embeddings generated. Shape: {self.z_all.shape}")
            if not torch.isfinite(self.z_all).all():
                 print("Warning: Non-finite values detected in final embeddings.")

        except RuntimeError as e:
             if "CUDA out of memory" in str(e):
                  print("\nFATAL ERROR: CUDA out of memory even with batched inference.")
                  print(f"Try reducing INFERENCE_BATCH_SIZE (current: {INFERENCE_BATCH_SIZE}).")
             else:
                  print(f"\nRuntime error during batched embedding generation: {e}")
             sys.exit()
        except Exception as e:
             print(f"Error generating final embeddings: {e}")
             sys.exit()

        print("\n--- Recommender Ready ---")

    def _get_query_embedding(self, text):
        """Helper to get the initial SciBERT embedding for a query text."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.feature_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().cpu()

    def recommend(self, query_title, query_abstract, k=10):
        """
        Recommends the top-k papers based on cosine similarity between the query's
        INITIAL embedding and the DATABASE papers' INITIAL embeddings (SciBERT content similarity).
        """
        if self.x_all is None or self.x_all.shape[0] == 0:
            print("Error: Initial embeddings (x_all) were not loaded or are empty.")
            return pd.DataFrame()

        # 1. Get the initial embedding for the query paper (on CPU)
        query_text = query_title + " " + query_abstract
        x_query = self._get_query_embedding(query_text)

        # 2. Map query embedding into the GAT latent space and compute similarity
        # Create a self-loop edge for a single-node graph so GATConv can operate
        print("Calculating similarities (graph-aware space)...")
        if x_query.shape[0] != 1:
             print(f"Error: Query embedding has unexpected shape: {x_query.shape}")
             return pd.DataFrame()

        edge_index_query = torch.tensor([[0],[0]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            z_query = self.encoder(x_query.to(self.device), edge_index_query).cpu()

        # Compare the query's graph-aware embedding against graph-aware embeddings of all papers
        sim = F.cosine_similarity(z_query, self.z_all)

        # 3. Get top-k recommendations
        print(f"Finding top {k} recommendations...")
        actual_k = min(k, sim.shape[0])
        if actual_k <= 0:
            print("Error: No similarity scores available.")
            return pd.DataFrame()
        if actual_k < k:
             print(f"Warning: Requested k={k} but only {actual_k} papers available. Returning {actual_k}.")

        # Handle potential NaNs in similarity scores before topk
        if not torch.isfinite(sim).all():
             print("Warning: Non-finite similarity scores detected. Replacing with -1.")
             sim = torch.nan_to_num(sim, nan=-1.0, posinf=-1.0, neginf=-1.0)

        top_k_scores, top_k_indices = torch.topk(sim, actual_k)

        # 4. Look up metadata
        results_data = []
        indices_np = top_k_indices.cpu().numpy()
        scores_np = top_k_scores.cpu().numpy()
        for i in range(actual_k):
             idx = int(indices_np[i]) # Convert numpy int64 to Python int
             score = scores_np[i]
             metadata = self.node_idx_to_metadata.get(idx, {})
             results_data.append({
                  'node_idx': idx,
                  'title': metadata.get('title', 'N/A'),
                  'year': metadata.get('year', 'N/A'),
                  'similarity_score': score
             })

        results_df = pd.DataFrame(results_data)
        return results_df[['node_idx', 'title', 'year', 'similarity_score']]

if __name__ == "__main__":
    recommender = Recommender()

    # --- DEMO ---
    demo_title = "Graph Attention Networks"
    demo_abstract = ("We present graph attention networks (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods' features, we enable (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront."
)

    print(f"\n--- Recommending citations for: '{demo_title}' ---")
    recommendations = recommender.recommend(demo_title, demo_abstract, k=10)

    print("\n--- Top 10 Recommendations ---")
    if not recommendations.empty:
        print(recommendations.to_string(index=False, float_format="%.4f"))
    else:
        print("No recommendations generated.")

