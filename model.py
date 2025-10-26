import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATEncoder(torch.nn.Module):
    """
    Graph Attention Network (GAT) Encoder.
    Takes node features and graph connectivity and returns
    final, graph-aware node embeddings.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        # On the last layer, we average the heads' outputs instead of concatenating
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, 
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        """
        Forward pass
        x: Node features [N, in_channels]
        edge_index: Graph connectivity [2, E]
        """
        # Apply dropout to input features
        x = F.dropout(x, p=0.6, training=self.training)
        
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Apply dropout
        x = F.dropout(x, p=0.6, training=self.training)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        
        return x # Final node embeddings [N, out_channels]

class LinkPredictor(torch.nn.Module):
    """
    MLP-based Link Predictor.
    Takes a pair of node embeddings (source and destination) and
    predicts the probability of a link between them.
    """
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()
        # We concatenate the two node embeddings, so in_channels is doubled
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, z, edge_index):
        """
        Forward pass
        z: All node embeddings [N, in_channels]
        edge_index: The specific edges to predict [2, num_edges_to_predict]
        """
        # Get the embeddings for the source nodes of each edge
        z_src = z[edge_index[0]]
        # Get the embeddings for the destination nodes of each edge
        z_dst = z[edge_index[1]]
        
        # Concatenate source and destination embeddings
        x = torch.cat([z_src, z_dst], dim=-1)
        
        # Pass through the MLP
        return self.mlp(x)
