## Scholarly-GAT: A Graph Attention Network for Link Prediction and Recommendation in Temporal Citation Networks
### citation-network-gnn

A complete, end-to-end pipeline for citation recommendation on the DBLP graph using Graph Neural Networks (GNNs). We:

- preprocess a large JSON dump into a graph (papers as nodes, citations as edges)
- generate dense text features for each paper from its title+abstract using a pretrained Transformer
- train a Graph Attention Network (GAT) encoder for temporal link prediction
- use the trained encoder to power a content-in-context citation recommender

This README explains, in detail, what each file does and why, plus exact commands to run each step and key gotchas.

### What you build
- **Graph**: Nodes = papers, Edges = citations.
- **Node features**: Transformer embeddings of title+abstract.
- **Encoder**: A 2-layer GAT that produces graph-aware embeddings.
- **Predictor**: An MLP trained for link prediction with negative sampling.
- **Recommender**: Ranks existing papers for a new query paper by cosine similarity between query text embedding and graph-aware embeddings of all papers.

---

## Setup

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2) Install dependencies
- The base dependencies are in `requirement.txt`.
- Installing PyTorch and PyTorch Geometric (PyG) sometimes requires version-specific wheels.

Recommended approach:
- Install PyTorch first following the official selector (choose CPU/GPU): [PyTorch install guide](https://pytorch.org/get-started/locally/)
- Then install PyG following: [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- Finally, install the rest:
```bash
pip install -r requirement.txt
```

Notes:
- If you installed a CUDA-enabled PyTorch, match the PyG wheels to the same torch and CUDA versions per the PyG guide.
- If you prefer CPU-only, install CPU builds of torch and PyG.

---

## Data acquisition

We use the DBLP JSON included in the Kaggle dataset: [mathurinache/citation-network-dataset](https://www.kaggle.com/datasets/mathurinache/citation-network-dataset).

1) Ensure you have a Kaggle API token at `~/.kaggle/kaggle.json` (Account → Settings → Create New API Token)
2) Run the downloader (it will install the `kaggle` package if missing):
```bash
python data/download_dataset.py
```
This produces `data/dblp-v12.json`.

---

## End-to-end pipeline (commands)

1) Preprocess the raw JSON into CSVs
```bash
python 01_preprocess_data.py
```
Outputs:
- `data/nodes.csv` with columns `[paper_id, year, title, abstract]`
- `data/edges.csv` with columns `[source_id, target_id]`

2) Generate Transformer features for each paper
```bash
python 02_generate_features.py
```
Outputs:
- `data/node_features.pt` (tensor of shape `[num_nodes, feature_dim]`)
- `data/paper_id_to_node_idx.json` (string `paper_id` → int `node_idx` mapping)

3) Train the GAT encoder + link predictor
```bash
python 03_train.py
```
Outputs:
- `data/gat_encoder.pt`
- `data/gat_predictor.pt`
- Console logs with Loss, Validation AUC/AP, and final Test AUC/AP

4) Run the recommender demo
```bash
python recommender.py
```
Outputs:
- Top‑k recommended papers for the included GAT demo abstract, with similarity scores.

---

## File-by-file deep dive (what and why)

### requirement.txt
Lists Python packages used by the project:
- **torch/torchvision/torchaudio**: Core PyTorch stack for tensor compute and training.
- **torch-geometric**: GNN layers/utilities; we use `GATConv` and negative sampling.
- **transformers**: Hugging Face models/tokenizers to embed titles+abstracts.
- **pandas**: CSV I/O and data manipulation.
- **tqdm**: Progress bars for long-running loops.
- **ijson**: Streaming JSON parser to iterate a massive JSON without loading it fully.
- **scikit-learn/scipy**: Metrics (ROC-AUC, Average Precision) and general utilities.

Why: This combination supports text embedding (content signal), GNN encoding (graph signal), and robust training/evaluation.

### data/download_dataset.py
Purpose: Automates fetching and extracting `dblp-v12.json` via Kaggle CLI into `data/`.

What it does:
- Verifies the dataset isn’t already extracted at `data/dblp-v12.json` to avoid duplicate work.
- Checks for `~/.kaggle/kaggle.json` and prints clear instructions if missing.
- Invokes `kaggle datasets download -d mathurinache/citation-network-dataset -p data/`.
- Unzips the downloaded archive into `data/` and removes the zip afterward.
- If run as a script, attempts to install `kaggle` automatically when missing.

Why: Keeps the dataset acquisition reproducible and non-interactive, which is important for fresh environments.

### 01_preprocess_data.py
Purpose: Stream the large DBLP JSON and turn it into two flat CSVs suitable for graph construction and temporal splits.

What it does:
- Uses `ijson.items(f_in, 'item')` to iterate paper objects without loading the whole JSON into memory.
- Filters out entries missing `id` or `year`, and skips invalid years.
- Writes `data/nodes.csv` with `[paper_id, year, title, abstract]`.
- Writes `data/edges.csv` with `[source_id, target_id]` for each reference in `paper['references']`.
- Tracks and prints the number of nodes and edges processed.

Why: The source JSON is massive; streaming keeps memory use bounded. We also enforce the presence of a valid year to enable a clean temporal train/val/test split later.

### 02_generate_features.py
Purpose: Convert each paper’s title+abstract into a fixed-size dense vector using a pretrained Transformer.

What it does:
- Loads `data/nodes.csv`, fills NAs, and constructs a contiguous integer index `node_idx` in memory (via `reset_index`).
- Builds and saves a persistent map `paper_id → node_idx` to `data/paper_id_to_node_idx.json`.
- Loads the tokenizer and model (`distilbert-base-uncased` by default; 768‑dim CLS embeddings) and moves the model to the available device (GPU if present).
- Tokenizes and encodes texts in batches to control VRAM/CPU memory usage.
- Extracts the `[CLS]` embedding (`outputs.last_hidden_state[:, 0, :]`) per paper and concatenates all batches into a single tensor `x`.
- Saves `x` to `data/node_features.pt`.

Why: The GNN needs initial node features. Using a strong text encoder gives a content-based signal for each paper before graph message passing. Persisting the `paper_id → node_idx` map ensures consistent index alignment across steps.

Model/Dim choices:
- `distilbert-base-uncased` → 768 dims (fast and decent quality).
- You can switch to `allenai/scibert_scivocab_uncased` (768 dims, domain-specific) or `sentence-transformers/all-MiniLM-L6-v2` (384 dims, very fast). If you change the feature dimension, update `IN_CHANNELS` in both `03_train.py` and `recommender.py` accordingly.

### model.py
Purpose: Define the learnable components used during training and inference.

What it defines:
- **GATEncoder**: A 2-layer Graph Attention Network.
  - Layer 1: `GATConv(in_channels, hidden_channels, heads=GAT_HEADS, dropout=0.6)` followed by `ELU` and dropout.
  - Layer 2: `GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)` (averaged heads).
  - Output: Final graph-aware node embeddings `z ∈ ℝ^{N×out_channels}`.
- **LinkPredictor**: A simple MLP over the concatenation `[z_src; z_dst]` for each edge.
  - Architecture: `Linear(2*in_channels → hidden) → ReLU → Dropout → Linear(hidden → 1)`
  - Output: A single logit per candidate edge.

Why:
- GAT uses attention to weigh neighbors, often improving over plain GCNs on citation graphs.
- The MLP link predictor is a standard, expressive yet simple baseline for link prediction.

### 03_train.py
Purpose: Train the GAT encoder + MLP predictor for temporal link prediction and evaluate with ROC‑AUC and Average Precision.

What it does:
1) Loads artifacts from previous steps:
   - `data/node_features.pt` as initial features `x` (shape `[N, feature_dim]`).
   - `data/paper_id_to_node_idx.json` as the canonical mapping from raw paper IDs to contiguous indices.
   - `data/nodes.csv` and `data/edges.csv` for metadata and edges.
2) Builds a `node_idx → year` map and converts `edges_df` from string IDs to integer indices using the saved mapping.
3) Performs a temporal split based on the source node’s year:
   - Train edges: source year ≤ `TRAIN_YEAR_END`
   - Val edges: source year = `VAL_YEAR`
   - Test edges: source year = `TEST_YEAR`
4) Initializes models (GAT encoder and MLP predictor), optimizer, and loss (BCEWithLogitsLoss).
5) Training loop per epoch:
   - Computes graph-aware embeddings `z = encoder(x, train_edge_index)` using only the training graph structure for message passing.
   - Uses the training edges as positive examples and samples an equal number of negatives via `negative_sampling`.
   - Concatenates positives and negatives, predicts logits with the predictor, computes BCE loss, and updates parameters.
   - Evaluates on validation edges (with negatives sampled from the training graph) using ROC‑AUC and AP.
6) After training: evaluates on the test set, then saves `gat_encoder.pt` and `gat_predictor.pt`.

Why:
- Temporal splitting simulates real‑world forecasting (recommend citations for future papers without leaking from the future).
- Negative sampling prevents a trivial predictor (most node pairs are non-edges) and makes training tractable.
- Evaluating with ROC‑AUC and AP provides robust ranking metrics for link prediction.

Important implementation note (node_idx mapping):
- `nodes.csv` does not store `node_idx` as a column. To build `node_idx → year`, reconstruct `node_idx` from `paper_id` using the saved JSON mapping. For example:
```python
nodes_df = pd.read_csv(NODES_FILE)
with open(NODE_MAP_FILE, 'r') as f:
    paper_id_to_node_idx = json.load(f)
nodes_df['node_idx'] = nodes_df['paper_id'].map(paper_id_to_node_idx)
node_idx_to_year = pd.Series(nodes_df.year.values, index=nodes_df.node_idx).to_dict()
```

### recommender.py
Purpose: Serve recommendations for a new query paper.

What it does:
- Loads the same feature model used in Step 02 (defaults to `distilbert-base-uncased`) for embedding queries at inference time.
- Loads the trained `gat_encoder.pt` and sets it to eval mode.
- Loads `nodes.csv`, `node_features.pt`, and `paper_id_to_node_idx.json` and rebuilds the training graph (`edge_index`) using the same temporal cutoff as training (`TRAIN_YEAR_END`).
- Computes graph-aware embeddings once for all papers: `z_all = encoder(x_all, train_edge_index)` and caches them in memory.
- For a new query paper (title+abstract), computes its initial text embedding with the same Transformer (`x_query`).
- Ranks existing papers by cosine similarity between `x_query` and each row of `z_all`, and returns the top‑k results with titles, years, and scores.

Why:
- Comparing the query’s raw content against the database’s graph-aware embeddings leverages both content and context: each database node embedding has already aggregated information from its citation neighborhood via attention.

Implementation detail (node_idx again):
- As in training, reconstruct `node_idx` from `paper_id` when building `node_idx → year` for the temporal filter.

---

## Data artifacts produced
- `data/dblp-v12.json`: Raw dataset from Kaggle (JSON array of paper objects).
- `data/nodes.csv`: Node table `[paper_id, year, title, abstract]`.
- `data/edges.csv`: Edge table `[source_id, target_id]` (citations).
- `data/paper_id_to_node_idx.json`: Persistent mapping `paper_id → node_idx`.
- `data/node_features.pt`: Tensor of size `[num_nodes, feature_dim]` with initial text features.
- `data/gat_encoder.pt`, `data/gat_predictor.pt`: Trained model weights.

---

## Configuration knobs to be aware of
- In `02_generate_features.py`:
  - **MODEL_NAME**: switch encoders; update downstream feature dims accordingly.
  - **BATCH_SIZE**: reduce if you hit OOM or increase for speed if you have VRAM.
- In `03_train.py`:
  - **TRAIN_YEAR_END / VAL_YEAR / TEST_YEAR**: temporal split.
  - **IN_CHANNELS / HIDDEN_CHANNELS / OUT_CHANNELS / GAT_HEADS**: GAT size.
  - **LEARNING_RATE / EPOCHS**: training schedule.
- In `recommender.py`:
  - **FEATURE_MODEL_NAME** and dims must match Step 02.
  - **TRAIN_YEAR_END** must match training.

---

## Troubleshooting and gotchas
- **PyTorch Geometric install errors**: Ensure torch/torchvision/torchaudio and PyG wheels match (version and CUDA). Use the official selectors from the linked guides above.
- **`nodes_df.node_idx` KeyError**: `nodes.csv` does not include `node_idx`. Always reconstruct via `paper_id_to_node_idx.json` and attach it as a new column before building `node_idx → year`.
- **Feature dimension mismatch**: If you change the Transformer, update `IN_CHANNELS` in both training and recommender to the correct embedding width (e.g., 768 for DistilBERT/SCIBERT, 384 for MiniLM L6).
- **Out of memory (GPU/CPU)**: Lower `BATCH_SIZE` in Step 02, and/or set `max_length=256` in tokenization. For training, reduce model sizes or batch negatives equal to positives (as done).
- **Hugging Face model downloads offline**: Set environment variable `TRANSFORMERS_CACHE` or ensure the environment has internet access for model weights.
- **Kaggle CLI not found**: `pip install kaggle` and verify `~/.kaggle/kaggle.json` exists and is readable.

---

## Example: programmatic use of the recommender
```python
from recommender import Recommender

recommender = Recommender()
title = "Graph Attention Networks"
abstract = "We present graph attention networks (GATs)..."
topk = recommender.recommend(title, abstract, k=10)
print(topk)
```

---

## Reproducibility notes
- Seeds are not set; expect slight variation in training metrics.
- All temporal splits are defined by source node year to avoid future leakage into training.
- Negative sampling is done each evaluation as well; metrics can vary slightly across runs.

---

## Project status
The pipeline runs end-to-end. Consider extending with:
- Mean pooling or `sentence-transformers` encoders instead of CLS token.
- Edge directionality variants or symmetric training.
- Hard negative mining.
- Exporting a `nodes_with_idx.csv` to avoid reconstructing `node_idx` repeatedly.
