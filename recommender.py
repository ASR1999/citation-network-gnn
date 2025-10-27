import torch
import torch.nn.functional as F
import pandas as pd
import json
import os
# Try importing safetensors, handle if not installed
try:
    import safetensors
    SAFE_TENSORS_AVAILABLE = True
except ImportError:
    SAFE_TENSORS_AVAILABLE = False
    print("Warning: 'safetensors' library not found. Falling back to default PyTorch loading.")
    print("Install with: pip install safetensors")

from transformers import AutoTokenizer, AutoModel
import sys
import numpy as np # Needed for numpy arrays in tensor creation

# Import our model class
from model import GATEncoder # Make sure model.py is in the same directory

# --- Configuration ---
DATA_DIR = 'data'
NODES_FILE = os.path.join(DATA_DIR, 'nodes.csv')
EDGES_FILE = os.path.join(DATA_DIR, 'edges.csv') # Needed for train_edge_index
NODE_FEATURES_FILE = os.path.join(DATA_DIR, 'node_features.pt')
NODE_MAP_FILE = os.path.join(DATA_DIR, 'paper_id_to_node_idx.json')

# --- Select which epoch's checkpoint to load ---
# Set this to the epoch number you want to use (e.g., the best one)
LOAD_EPOCH = 1 # Change this based on your training results
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')
ENCODER_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'gat_encoder_epoch_{LOAD_EPOCH:02d}.pt')
# PREDICTOR_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'gat_predictor_epoch_{LOAD_EPOCH:02d}.pt') # Predictor not strictly needed for this type of recommendation

# --- MUST match the models used in TRAINING (03_train.py) ---
FEATURE_MODEL_NAME = 'allenai/scibert_scivocab_uncased' # Changed to SciBERT
IN_CHANNELS = 768
HIDDEN_CHANNELS = 256 # Corrected
OUT_CHANNELS = 256   # Corrected
GAT_HEADS = 8        # Corrected
TRAIN_YEAR_END = 2017 # Must match 03_train.py
# ---------------------

class Recommender:
    def __init__(self):
        print("Initializing recommender...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Load Feature model (SciBERT) for embedding queries
        print(f"Loading feature model: {FEATURE_MODEL_NAME}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(FEATURE_MODEL_NAME)
            # *** Add use_safetensors=True if available ***
            load_args = {'use_safetensors': True} if SAFE_TENSORS_AVAILABLE else {}
            print(f"Attempting to load feature model with args: {load_args}")
            self.feature_model = AutoModel.from_pretrained(FEATURE_MODEL_NAME, **load_args).to(self.device)
            self.feature_model.eval()
        except RuntimeError as e:
            # Catch the specific security error again
            if "upgrade torch to at least v2.6" in str(e):
                 print("\nFATAL ERROR: Your PyTorch version is too old and insecure for torch.load.")
                 print("The 'safetensors' workaround did not prevent the error.")
                 print("The only remaining option is to create a new environment with Python 3.11/3.12 and the latest PyTorch/PyG.")
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
            # Use weights_only=True for safety when loading state dicts
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
            # Ensure node_idx exists
            if 'node_idx' not in self.nodes_df.columns:
                 self.nodes_df = self.nodes_df.reset_index().rename(columns={'index': 'node_idx'})

            with open(NODE_MAP_FILE, 'r') as f:
                self.paper_id_to_node_idx = json.load(f)
            # Create reverse map for convenience
            self.node_idx_to_metadata = self.nodes_df.set_index('node_idx').to_dict('index')

        except FileNotFoundError as e:
            print(f"Error: Required data file not found: {e}")
            sys.exit()
        except Exception as e:
            print(f"Error loading metadata/map: {e}")
            sys.exit()


        # 4. Load all initial features
        print(f"Loading initial node features from {NODE_FEATURES_FILE}...")
        try:
             # Load features to CPU first, then move if needed, prevents large GPU mem spike
             self.x_all = torch.load(NODE_FEATURES_FILE, map_location='cpu', weights_only=True)
             print(f"Initial features shape: {self.x_all.shape}")
        except FileNotFoundError:
            print(f"Error: Node features file not found at {NODE_FEATURES_FILE}")
            sys.exit()
        except RuntimeError as e:
             # Catch the specific security error if it still occurs
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
            # Use .get() for safer dictionary access, provide default (e.g., 0 or np.nan)
            node_idx_to_year = pd.Series(self.nodes_df.year.values, index=self.nodes_df.node_idx).to_dict()

            edges_df = pd.read_csv(EDGES_FILE, dtype={'source_id': str, 'target_id': str})
            edges_df['source_idx'] = edges_df['source_id'].map(self.paper_id_to_node_idx)
            edges_df['target_idx'] = edges_df['target_id'].map(self.paper_id_to_node_idx)
            edges_df = edges_df.dropna(subset=['source_idx', 'target_idx'])
            edges_df['source_idx'] = edges_df['source_idx'].astype(int)
            edges_df['target_idx'] = edges_df['target_idx'].astype(int)
             # Use .get() for mapping years, handling missing keys
            edges_df['source_year'] = edges_df['source_idx'].apply(lambda idx: node_idx_to_year.get(idx, 0)) # Default to 0 if not found
            edges_df = edges_df.dropna(subset=['source_year']) # Drop if still NaN after get
            edges_df['source_year'] = edges_df['source_year'].astype(int)

            train_edges_df = edges_df[(edges_df['source_year'] > 0) & (edges_df['source_year'] <= TRAIN_YEAR_END)]

            # Convert numpy arrays to tensor directly
            train_edge_index_np = np.stack([train_edges_df['source_idx'].values,
                                            train_edges_df['target_idx'].values])
            self.train_edge_index = torch.tensor(train_edge_index_np, dtype=torch.long) # Keep on CPU for now

            print(f"Training graph edge_index shape: {self.train_edge_index.shape}")

        except FileNotFoundError as e:
             print(f"Error: Edges file not found: {e}")
             sys.exit()
        except Exception as e:
             print(f"Error processing edges: {e}")
             sys.exit()


        # 6. Generate final, graph-aware embeddings FOR ALL PAPERS
        # This is a one-time calculation during initialization
        print("Generating final graph-aware embeddings for all papers (this takes time and VRAM)...")
        self.z_all = None # Initialize z_all
        try:
            with torch.no_grad():
                # Move necessary data to GPU only for this calculation
                x_all_gpu = self.x_all.to(self.device)
                train_edge_index_gpu = self.train_edge_index.to(self.device)
                # Ensure encoder output is detached before moving to CPU
                self.z_all = self.encoder(x_all_gpu, train_edge_index_gpu).detach().cpu() # Calculate on GPU, store on CPU
                del x_all_gpu # Free GPU memory
                del train_edge_index_gpu
                torch.cuda.empty_cache()
            print(f"Final embeddings generated. Shape: {self.z_all.shape}")
        except RuntimeError as e:
             if "CUDA out of memory" in str(e):
                  print("\nFATAL ERROR: CUDA out of memory while generating final embeddings.")
                  print("Your GPU may not have enough VRAM to process the entire graph at once.")
                  print("Consider reducing model size or implementing batch processing for inference.")
             else:
                  print(f"\nRuntime error during final embedding generation: {e}")
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

        # Use [CLS] token embedding, move to CPU
        return outputs.last_hidden_state[:, 0, :].detach().cpu() # Detach before moving

    def recommend(self, query_title, query_abstract, k=10):
        """
        Recommends the top-k papers based on cosine similarity between the query's
        INITIAL embedding and the DATABASE papers' FINAL (graph-aware) embeddings.
        """
        if self.z_all is None:
            print("Error: Final embeddings (z_all) were not generated.")
            return pd.DataFrame() # Return empty dataframe

        # 1. Get the initial embedding for the query paper (on CPU)
        query_text = query_title + " " + query_abstract
        x_query = self._get_query_embedding(query_text)

        # 2. Compute similarity (all tensors should be on CPU now)
        # Compare the query's initial embedding (x_query) against
        # the graph-aware embeddings of all papers in the database (z_all).
        print("Calculating similarities...")
        # Ensure dimensions match before cosine similarity
        if x_query.shape[0] != 1 or self.z_all.shape[0] == 0:
             print(f"Error: Embedding dimension mismatch or empty database embeddings.")
             print(f"Query shape: {x_query.shape}, Database shape: {self.z_all.shape}")
             return pd.DataFrame()
        # Cosine similarity expects input B N D and comparison N D -> returns B N
        # We have query 1 D and database M D -> should return 1 M
        sim = F.cosine_similarity(x_query, self.z_all) # This computes similarity row-wise if dimensions > 1

        # 3. Get top-k recommendations
        print(f"Finding top {k} recommendations...")
        # Ensure k is not larger than the number of papers
        actual_k = min(k, sim.shape[0])
        if actual_k < k:
             print(f"Warning: Requested k={k} but only {actual_k} papers available. Returning {actual_k}.")
        if actual_k == 0:
             print("Error: No similarity scores available to find top recommendations.")
             return pd.DataFrame()

        top_k_scores, top_k_indices = torch.topk(sim, actual_k)

        # 4. Look up metadata
        # Use the stored node_idx_to_metadata dictionary for faster lookup
        results_data = []
        for idx, score in zip(top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy()): # Ensure indices/scores are on CPU
             # idx is the index in the z_all tensor, which corresponds to node_idx
             metadata = self.node_idx_to_metadata.get(int(idx), {}) # Convert idx to int for dict lookup
             results_data.append({
                  'node_idx': int(idx),
                  'title': metadata.get('title', 'N/A'),
                  'year': metadata.get('year', 'N/A'),
                  'similarity_score': score
             })

        results_df = pd.DataFrame(results_data)
        return results_df[['node_idx', 'title', 'year', 'similarity_score']]

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

    print("\n--- Top 10 Recommendations ---")
    # Improve formatting for readability
    if not recommendations.empty:
        print(recommendations.to_string(index=False, float_format="%.4f"))
    else:
        print("No recommendations generated.")

