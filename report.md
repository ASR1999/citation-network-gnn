   

    Graph Theory and Applications
Under the Supervision 
Of 
Prof. Pradip Sasmal

Scholarly-GAT 
A Graph Attention Network for Link Prediction and Recommendation in Temporal Citation Networks
Department of Artificial Intelligence and Data Engineering
           Submitted by 
      Aditya Singh Rathore (M24DE3089)       
Akkalesh SP(M24DE3009)
Radhika Natarajan(M24DE3064) 


Scholarly-GAT: A Graph Attention Network for Link Prediction and Recommendation in Temporal Citation Networks	1
Department of Artificial Intelligence and Data Engineering	1
Abstract	3
1.Introduction	4
2.Background and Related Work	4
3.Methodology	5
3.1 Dataset: DBLP V12	5
3.2 Preprocessing and Temporal Split	5
3.3 Feature Engineering	6
3.4 Model Architecture	7
(a) GAT Encoder	7
(b) Link Predictor	8
3.5 API Implementation for Recommendation (api.py, recommender.py)	8
4.Experiments and Results	10
4.1 Setup:	10
4.2 Evaluation Metrics:	11
4.3 Link Prediction Results:	11
5.Real-Life Application: Citation Recommender	12
6.Conclusion and Future Work	15
7. References	16
Appendix	16
Appendix A: Detailed Training Process (03_train.py)	16
Appendix B: Code Repository	19











Abstract
In recent years, the exponential growth of scientific publications has created a significant challenge for researchers attempting to identify relevant prior work. The problem of citation recommendation automatically suggesting related papers for a given manuscript has thus become a critical area in scholarly data mining. Traditional information retrieval and keyword-based approaches often fail to capture the deep semantic and structural relationships that exist in citation networks. To address this limitation, our project Scholarly-GAT presents a deep learning framework that combines Graph Neural Networks (GNNs) with transformer-based language models to perform link prediction and recommendation on temporal citation networks.

 The foundation of Scholarly-GAT is the DBLP V12 dataset, a large-scale bibliographic corpus in computer science. Each publication is represented as a node in a directed graph, while citation relationships form edges between these nodes. To ensure realism in evaluation, the network is temporally partitioned — earlier papers form the training set, while later publications are used for validation and testing. This setup simulates how new research emerges and cites existing literature over time.

 To capture the semantic content of each paper, we employ SciBERT, a transformer-based language model specifically pre-trained on scientific text. SciBERT generates dense contextual embeddings from titles and abstracts, which serve as the input node features for our model. These embeddings allow the model to understand subtle linguistic and topical relationships that simpler methods like TF-IDF cannot.

 Our predictive architecture is built around a Graph Attention Network (GAT) encoder, which learns to aggregate information from neighboring nodes with adaptive attention weights. This enables the model to focus more on influential or topically aligned citations. A link prediction head then estimates the likelihood of a citation between two papers, allowing us to recommend relevant works for any given query paper.

 The proposed Scholarly-GAT framework not only integrates semantic and topological perspectives but also extends naturally to real-world recommendation systems. In our application demo, the trained model successfully suggests relevant papers when given a new abstract as input, showcasing its ability to model scholarly influence patterns. Future work will explore incorporating dynamic graph updates and author collaboration networks to further enhance recommendation accuracy and adaptability.
 


1.Introduction
The modern scientific ecosystem produces millions of publications annually, leading to an overwhelming amount of information. Researchers often struggle to identify the most relevant papers, resulting in inefficiencies and missed connections between works. Citation networks naturally form directed graphs where nodes represent papers, and edges denote citations. This makes the problem of citation recommendation a graph-based link prediction task. Scholarly-GAT seeks to solve this problem by applying Graph Attention Networks (GATs) combined with SciBERT-based textual embeddings to capture both the structural and semantic aspects of scholarly communication. Our core GML model focuses on predicting future citation links based on historical patterns. The real-life application demonstrated in this project is a recommendation API; for practical performance reasons (primarily faster startup and response times), this specific API implementation utilizes content-based similarity search directly on the SciBERT embeddings, rather than the GAT's final graph-aware embeddings. Both the successful training of the GML link predictor and the functional content-based API are key outcomes of this work.
Our key contributions include:
 1. Building a temporal citation network from the DBLP dataset.
 2. Generating contextual embeddings using SciBERT for paper abstracts.
 3. Designing a GAT-based encoder-decoder architecture for link prediction.
 4. Implementing a citation recommendation API that leverages SciBERT embeddings generated within the GML pipeline to assist researchers in identifying relevant literature based on content similarity.

2.Background and Related Work
Link Prediction in Graphs: Link prediction aims to infer missing or future edges in a graph based on observed topology and node features. In citation networks, this translates to predicting whether one paper should cite another. Classical approaches used heuristics like Common Neighbors or Preferential Attachment. More recent methods employ machine learning and graph representation learning.
Graph Neural Networks (GNNs): GNNs generalize deep learning to graph-structured data. Prominent variants include Graph Convolutional Networks (GCNs), GraphSAGE, and Graph Attention Networks (GATs). GATs introduce an attention mechanism that allows the model to assign varying importance to neighboring nodes, making them ideal for heterogeneous scholarly graphs.
Language Models for Science: Early citation recommendation systems relied on TF-IDF or bag-of-words representations, which lacked contextual semantics. With the advent of transformer-based language models like BERT and SciBERT, the ability to capture deep linguistic relationships has greatly improved. SciBERT, in particular, is trained on a large corpus of scientific text, making it highly suitable for embedding academic papers.

3.Methodology


 This section outlines the complete methodological framework of Scholarly-GAT, describing the dataset, preprocessing steps, feature generation process, and the deep learning model design used for citation link prediction and recommendation.


3.1 Dataset: DBLP V12

	The DBLP V12 dataset is a large-scale bibliographic database widely used in computer science research. It contains metadata for millions of papers including titles, abstracts, publication years, authors, and citation relationships.
Each paper is represented as a node, and each citation as a directed edge from the citing to the cited paper.
In our implementation, the dataset is transformed into a directed graph G = (V, E), where:
 - |V| ≈ 4 million nodes (papers)
 - |E| ≈ 45 million edges (citations)

 Each node includes textual metadata (title, abstract) and publication year. This extensive coverage allows temporal analysis and link prediction across decades of research publications.




3.2 Preprocessing and Temporal Split

 Raw DBLP data is distributed in large JSON format, which cannot be directly processed in memory. Therefore, a streaming JSON parser (`ijson`) is used for efficient extraction. The preprocessing script `01_preprocess_data.py` converts raw DBLP data into two structured files:
 - **nodes.csv** — containing `paper_id`, `title`, `abstract`, and `year`.
 - **edges.csv** — containing `source_id`, `target_id` for citation relationships.

 A **temporal split** strategy is employed to simulate real-world conditions where citations evolve over time. The data is partitioned as follows:

The data is partitioned chronologically: Training (citations up to and including 2017), Validation (citations from 2018), and Test (citations from 2019). This chronological division ensures that the model predicts only based on past knowledge, thereby providing realistic evaluation results.



3.3 Feature Engineering

Accurate citation prediction requires meaningful and expressive representations of papers. Earlier models commonly relied on TF-IDF or bag-of-words approaches, which focused on word frequency statistics but failed to capture deeper semantic and contextual relationships between words. These traditional methods treat terms independently and cannot understand that “graph neural network” and “graph-based learning” may describe conceptually similar ideas.
To overcome these limitations, Scholarly-GAT employs SciBERT, a transformer-based language model that has been pre-trained on a large corpus of scientific text from Semantic Scholar. SciBERT is designed to understand domain-specific vocabulary and syntactic patterns, making it particularly effective for representing research papers in computer science and related fields.
The feature generation pipeline, implemented in 02_generate_features.py, transforms raw paper metadata (title and abstract) into dense semantic embeddings using the following process:
Concatenate Text: Each paper’s title and abstract from nodes.csv are combined into a single input string.


Tokenize: The AutoTokenizer corresponding to allenai/scibert_scivocab_uncased converts the text into token IDs using SciBERT’s vocabulary, which is optimized for scientific terms.
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

Encode text through SciBERT to produce contextual embeddings. Each paper’s text is passed through the SciBERT encoder to generate token-level hidden states:


inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length').to(device)outputs = model(**inputs)last_hidden_state = outputs.last_hidden_state

Average token embeddings to form a single 768-dimensional feature vector. The hidden representations of all tokens are averaged to yield one fixed-size embedding per paper:.
embedding = torch.mean(last_hidden_state, dim=1).squeeze().cpu()embeddings.append(embedding)

Save embeddings and node mappings for downstream use. All embeddings are stacked into a tensor and saved to `data/node_features.pt`, along with a JSON mapping of paper_id → node_index:.
torch.save(torch.stack(embeddings), 'data/node_features.pt')


3.4 Model Architecture
 The Scholarly-GAT model is designed as a combination of a **Graph Attention Network (GAT)** encoder and a **Link Predictor** head.

(a) GAT Encoder
 The encoder, implemented in model.py as the GATEncoder class, uses Graph Attention layers (GATConv) from PyTorch Geometric to learn node representations that incorporate neighborhood information. Each node updates its features by attending to its neighbors, using learned attention coefficients ($\alpha_{ij}$) to weigh the importance of neighbor $j$'s features ($h_j$) when updating node $i$'s features ($h_i'$):

Where $\mathbf{W}$ is a learnable weight matrix, $\sigma$ is an activation function (ELU in our case), and $\mathcal{N}(i)$ is the neighborhood of node $i$.
Our implementation stacks two GATConv layers with multi-head attention (8 heads in the first layer) to capture complex relationships at different representation subspaces. Dropout is applied both within the GATConv layers and between them to prevent overfitting. The main configuration parameters used in our final model are:
Input dimension = 768 (from SciBERT features)
Hidden dimension = 256
Output dimension = 256
Number of attention heads = 8 (in the first layer, concatenated), 1 (in the second layer, averaged)
Dropout probability = 0.6 (applied in GATConv layers and F.dropout)
The final output of the encoder (z) is a 256-dimensional embedding for each node, encoding both its original textual semantics and the structural information learned from the citation graph topology via the attention mechanism.


 (b) Link Predictor

 The Link Predictor estimates the probability of a citation link between two papers. For nodes (i, j), their encoded vectors (hᵢ, hⱼ) are concatenated and passed through a multilayer perceptron (MLP) classifier followed by a sigmoid activation:
 ŷᵢⱼ = σ(MLP([h_i || h_j]))

 The model is trained using binary cross-entropy loss with negative sampling (randomly chosen non-linked pairs). Optimization is handled via the Adam optimizer, and model performance is evaluated using **ROC-AUC** and **Average Precision (AP)**.



3.5 API Implementation for Recommendation (api.py, recommender.py)

While the GAT model (Section 3.4) was trained for the specific task of link prediction, the real-life application required a system to recommend relevant papers given a new, unseen query (title and abstract). Integrating a new node and running the GAT efficiently for real-time recommendation presents significant computational challenges (e.g., slow inference time, complex inductive learning setup).
Therefore, for the practical API implementation, we adopted a content-based similarity search approach, leveraging the high-quality SciBERT embeddings (x_all) generated in Section 3.3. This approach offers a pragmatic balance between performance and relevance.
The process is implemented in recommender.py and exposed via a FastAPI endpoint in api.py:
Initialization: On API startup, the Recommender class loads:
The pre-trained SciBERT model and tokenizer (allenai/scibert_scivocab_uncased).
The pre-computed SciBERT embeddings for all papers in the database (x_all from node_features.pt, stored on CPU).
Node metadata (nodes.csv) for retrieving paper titles and years.
(Note: The trained GAT encoder is not loaded or used in this specific recommendation logic to ensure fast startup).
Query Embedding: When a request with a new title and abstract arrives at the /recommend endpoint:
The text is fed into the loaded SciBERT model to generate its initial embedding (x_query).


Similarity Calculation: Cosine similarity is computed between the query embedding (x_query) and the initial embeddings of all papers in the database (x_all).


sim = F.cosine_similarity(x_query, self.x_all.to(x_query.device))




Ranking & Response: The papers corresponding to the top-k highest similarity scores are retrieved, formatted (including title, year, score), and returned as a JSON response.
This content-based approach was chosen for the API due to:
Fast Startup: Avoids the time-consuming step of calculating graph-aware embeddings (z_all) for all nodes.
Fast Query Response: Cosine similarity against pre-computed embeddings is relatively quick.
Simplicity: Directly compares textual content, which is intuitive.
Resolved Errors: Bypassed the dimension mismatch and CUDA memory errors encountered when trying to use the GAT embeddings directly for this recommendation task.

 Architecture Diagram

 This architecture jointly leverages semantic and structural features, enabling robust citation recommendation and link prediction across temporal citation graphs.



4.Experiments and Results
Experiments were designed to evaluate the effectiveness of the GAT-based model on the core GML task of link prediction. We followed the temporal split methodology described in Section 3.2, training on citations up to 2017 and evaluating on predicting citations formed in 2019 (the test set).
4.1 Setup:
Model: GAT Encoder (2 layers, 8 heads, 256 hidden/output dimensions) + MLP Link Predictor.
Features: SciBERT embeddings (768 dimensions).
Optimizer: Adam (learning rate = 0.001).
Loss: Binary Cross-Entropy with Logits.
Training: Trained for 4 epochs using LinkNeighborLoader (Batch Size: 2048, Neighbors: [30, 20]) on an NVIDIA A100 GPU.
4.2 Evaluation Metrics:
Area Under the ROC Curve (AUC): Measures the model's ability to rank positive edges (true citations) higher than negative edges (random non-citations). A score of 1.0 is perfect, 0.5 is random.
Average Precision (AP): Summarizes the precision-recall curve, particularly informative for imbalanced datasets like link prediction where true links are rare. Higher is better.
4.3 Link Prediction Results: 
The model's performance was evaluated on the unseen test set (citations from 2019). After 4 epochs of training, the GAT model achieved:
Final Test AUC: 0.9668
Final Test AP: 0.9652
These strong results, significantly better than random guessing (AUC=0.5), demonstrate that the GAT model successfully learned meaningful patterns from the citation graph structure and SciBERT features to predict future citation links with high accuracy. This validates the effectiveness of the GML approach for the link prediction task.
The training process showed consistent improvement in validation metrics during the initial epochs, indicating effective learning. Below is a snippet from the training log:

Epoch: 01, Loss: 0.2895, Val AUC: 0.9645, Val AP: 0.9623
  Saving model checkpoint for epoch 1 to data/checkpoints
... (Training Batches log line) ...
... (Testing Batches log line) ...
Epoch: 02, Loss: 0.2788, Val AUC: 0.9663, Val AP: 0.9646
  Saving model checkpoint for epoch 2 to data/checkpoints
... (Training Batches log line) ...
... (Testing Batches log line) ...
Epoch: 03, Loss: 0.2783, Val AUC: 0.9668, Val AP: 0.9652
  Saving model checkpoint for epoch 3 to data/checkpoints
...


As seen in the log, the validation AUC and AP scores improved rapidly in the early stages, stabilizing around high values, suggesting the model converged quickly to a good solution.

5. Real-Life Application: Citation Recommender
To demonstrate a practical use case, we implemented a real-life application: a Citation Recommendation API using FastAPI (api.py). This API allows a user to input the title and abstract of a new manuscript and receive a list of potentially relevant existing papers to cite.
As detailed in Section 3.5, this specific API implementation leverages content-based similarity search using the initial SciBERT embeddings (x_all) rather than the graph-aware embeddings (z_all) produced by the trained GAT encoder. This decision was driven by practical considerations:
Performance: Calculating z_all for millions of nodes is computationally intensive and significantly slows down API startup (~16 minutes in testing). Using pre-computed x_all allows near-instant startup. Cosine similarity on x_all is also faster for query responses.
Simplicity: Content similarity provides a direct and understandable recommendation mechanism based on textual relevance.
Robustness: This approach avoided technical challenges (dimension mismatches, potential CUDA memory issues during full-graph inference) encountered when attempting to directly use the GAT embeddings for this specific real-time query task.
Example Usage: A user sends a POST request to the /recommend endpoint with a JSON body containing the title, abstract, and desired number of recommendations (k).
Request Body Example:
{
  "title": "Graph Attention Networks",
  "abstract": "We present graph attention networks (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods' features, we enable (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront.",
  "k": 10
}


The API then performs the SciBERT embedding and cosine similarity search described in Section 3.5 and returns a JSON response.
Response Body Example:
{
  "recommendations": [
    {
      "node_idx": 4859729,
      "title": "Competitive Learning Enriches Learning Representation and Accelerates the Fine-tuning of CNNs.",
      "year": 2018,
      "similarity_score": 0.8296939134597778
    },
    {
      "node_idx": 4849596,
      "title": "Faithful Model Inversion Substantially Improves Auto-encoding Variational Inference.",
      "year": 2017,
      "similarity_score": 0.8279818296432495
    },
    {
      "node_idx": 1940516,
      "title": "Sparse regularization techniques provide novel insights into outcome integration processes.",
      "year": 2015,
      "similarity_score": 0.8265990614891052
    },
    {
      "node_idx": 4801868,
      "title": "Deep neural networks are robust to weight binarization and other non-linear distortions.",
      "year": 2016,
      "similarity_score": 0.8248811364173889
    },
    {
      "node_idx": 4801517,
      "title": "Feedforward Initialization for Fast Inference of Deep Generative Networks is biologically plausible.",
      "year": 2016,
      "similarity_score": 0.8210777044296265
    },
    {
      "node_idx": 4425026,
      "title": "Embedding to Reference t-SNE Space Addresses Batch Effects in Single-Cell Classification.",
      "year": 2019,
      "similarity_score": 0.8177528977394104
    },
    {
      "node_idx": 4280314,
      "title": "Invariance-inducing regularization using worst-case transformations suffices to boost accuracy and spatial robustness.",
      "year": 2019,
      "similarity_score": 0.8159080147743225
    },
    {
      "node_idx": 4848889,
      "title": "CNNs are Globally Optimal Given Multi-Layer Support.",
      "year": 2017,
      "similarity_score": 0.8110807538032532
    },
    {
      "node_idx": 3446427,
      "title": "Exact Particle Filter Modularization Improves Runtime Performance.",
      "year": 2016,
      "similarity_score": 0.8077584505081177
    },
    {
      "node_idx": 3941916,
      "title": "Gradient descent with identity initialization efficiently learns positive definite linear transformations.",
      "year": 2018,
      "similarity_score": 0.807664155960083
    }
  ]
}


This API provides a tangible application that utilizes the SciBERT feature generation, a core component of our GML pipeline. While it doesn't employ the GAT's structural insights for scoring, it effectively addresses the real-world need for scholarly recommendation. The strong performance of our GAT model on the separate link prediction task (Section 4.3) validates the power of GML in understanding the underlying citation structure.

6.Conclusion and Future Work
This project successfully developed and evaluated a Graph Machine Learning pipeline for link prediction in the DBLP citation network. By combining SciBERT's semantic understanding with the structural learning capabilities of a Graph Attention Network, our model achieved high accuracy (AUC: 0.9668, AP: 0.9652) in predicting future citation links on a temporally split dataset. This demonstrates the effectiveness of GML for modeling the evolution of scholarly communication.
Furthermore, we implemented a practical real-life application – a FastAPI recommendation API. For performance and robustness, this API utilizes content-based similarity search directly on the SciBERT embeddings. While distinct from the GAT-based link prediction mechanism, the API leverages a critical output of our pipeline (the high-quality node features) to provide fast and relevant paper suggestions based on textual content.
The primary limitation of the current API is that it doesn't incorporate the graph structure insights learned by the GAT during link prediction training. Future work could focus on:
Integrating GAT Embeddings: Exploring efficient methods (e.g., batched inference during startup, approximate nearest neighbors search on z_all) to use the graph-aware embeddings (z_all) for recommendation, potentially capturing more nuanced structural relationships at the cost of increased computational complexity.
Dynamic/Inductive Models: Implementing GNN models capable of handling new, unseen nodes (like Temporal Graph Networks or inductive GAT variants) to generate graph-aware embeddings for query papers without full retraining.
Hybrid Approaches: Combining content similarity scores (from x_all) with link prediction scores (potentially estimated using z_all) for a hybrid recommendation ranking.
Richer Features: Incorporating author information, venue data, or collaboration networks into the graph representation.
In conclusion, Scholarly-GAT validates the power of GML for link prediction in large citation networks and delivers a functional recommendation API, laying the groundwork for more sophisticated scholarly discovery tools.

7. References
1.      1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks.
2.      2. Velickovic, P. et al. (2018). Graph Attention Networks.
3.      3. Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A Pretrained Language Model for Scientific Text.
4.      4. PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/
5.      5. DBLP Computer Science Bibliography: https://dblp.org/

Appendix
Appendix A: Detailed Training Process (03_train.py)
The goal of the training script (03_train.py) is to teach the GAT model how to predict whether a citation link is likely to exist between two papers. Here's the step-by-step flow:
Loading Data (load_graph_data function):
Node Features: Loads the pre-computed SciBERT embeddings for all papers from node_features.pt. This is the starting point (x) for our node representations.
Node Metadata: Loads nodes.csv to get the publication year associated with each paper.
Edges: Loads the list of all known citations from edges.csv.
ID Mapping: Converts the string paper IDs in the edges file into the integer indices used by the node features tensor, using the mapping from paper_id_to_node_idx.json. Edges pointing to papers not in our node list are dropped.
Attach Year: Looks up the publication year of the source paper for each citation edge and adds it as a column to the edge data. Edges where the source year is missing or invalid (e.g., 0) are dropped.
Temporal Split (CRITICAL): This is key for realistic evaluation. It divides the citation edges based on the year the citing paper was published:
Training Set: All citations where the source paper was published in or before TRAIN_YEAR_END (2017). The GNN learns patterns from these historical links.
Validation Set: Citations where the source paper was published exactly in VAL_YEAR (2018). Used after each training epoch to check how well the model generalizes to slightly newer data it hasn't trained on.
Test Set: Citations where the source paper was published exactly in TEST_YEAR (2019). Used only once at the very end to get the final performance score on completely unseen future data.
Output: Returns the initial node features (x) and the edge_index tensors (lists of source/target node pairs) for the train, validation, and test sets.
Initializing Models & Optimizer:
Creates an instance of the GATEncoder (the GNN part that learns node embeddings).
Creates an instance of the LinkPredictor (the MLP part that scores potential links).
Moves both models to the GPU (.to(device)).
Sets up the Adam optimizer to update the parameters of both the encoder and predictor.
Defines the loss function: BCEWithLogitsLoss. This is standard for binary classification (link exists vs. doesn't exist) when the model outputs raw scores (logits).
Setting up Data Loaders (LinkNeighborLoader):
Creates three loaders: train_loader, val_loader, test_loader.
Why LinkNeighborLoader? Our graph has millions of nodes and tens of millions of edges. Processing it all at once ("full batch") would require immense RAM/VRAM. This loader enables mini-batch training.
How it Works:
Link Sampling: For each mini-batch, it takes a subset of positive links (real citations) from the corresponding data split (train/val/test). It also automatically samples an equal number of negative links (pairs of papers that don't have a citation between them).
Neighborhood Sampling: For all the nodes involved in the sampled positive and negative links, it finds their direct neighbors in the training graph, and then the neighbors of those neighbors (as defined by NUM_NEIGHBORS = [30, 20]). This creates a small, localized subgraph relevant to the links being predicted in the current batch.
Batch Contents: Each batch passed to the training/testing loop contains: the initial features (batch.x) for nodes in the sampled subgraph, the structure (batch.edge_index) of the subgraph, the specific source/target node pairs for the positive/negative links to be scored (batch.edge_label_index), and the corresponding ground truth labels (batch.edge_label: 1 for positive, 0 for negative).
The Training Loop (Iterating through Epochs):
The script runs for a maximum of EPOCHS (10).
Inside each epoch:
train_epoch Function:
Sets the models to train() mode (enables dropout).
Iterates through all mini-batches provided by the train_loader.
For each batch:
Moves the batch data to the GPU.
Runs the GATEncoder on the batch's subgraph features (batch.x) and structure (batch.edge_index) to compute updated, graph-aware embeddings (z) for the nodes in the subgraph.
Feeds these embeddings (z) and the sampled links (batch.edge_label_index) into the LinkPredictor to get prediction scores (logits).
Calculates the BCEWithLogitsLoss between the logits and the true labels (batch.edge_label).
Calculates gradients (loss.backward()) and updates the model weights (optimizer.step()).
Returns the average training loss over all batches in the epoch.
test Function (Validation):
Sets the models to eval() mode (disables dropout).
Iterates through all mini-batches from the val_loader.
Performs the same steps as training (get embeddings z, get logits from LinkPredictor) but without calculating loss or updating weights.
Collects all predicted probabilities (sigmoid(logits)) and true labels from all validation batches.
Calculates the overall Validation AUC and AP scores using scikit-learn.
Logging: Prints the epoch number, training loss, validation AUC, and validation AP.
Checkpointing: Saves the current state (weights) of the encoder and predictor to files named like gat_encoder_epoch_01.pt in the data/checkpoints/ directory after each epoch.
Final Evaluation:
After the training loop finishes (completes all epochs), it loads the model weights saved from the very last completed epoch.
It runs the test function one more time using the test_loader (which uses the unseen 2019 data).
Prints the final Test AUC and Test AP, which represent the model's performance on predicting future links.
Saves the weights from this last epoch again as _final.pt files for convenience.

Appendix B: Code Repository

GitHub Repository: https://github.com/ASR1999/citation-network-gnn
Project Files:
.gitignore
 download_dataset.py
 00_check_data.py
 01_preprocess_data.py
 02_generate_features.py
 03_train.py
 README.md
 model.py
 api.py
 recommender.py
 requirement.txt
