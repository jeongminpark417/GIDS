import torch
import torch.nn.functional as F

# from torch_geometric.nn import HeteroConv, GATConv, GCNConv, SAGEConv
# from torch_geometric.utils import trim_to_layer

from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, HeteroGraphConv
from dgl import apply_each


class RGNN(torch.nn.Module):
#  r""" [Relational GNN model](https://arxiv.org/abs/1703.06103).

#   Args:
#     etypes: edge types.
#     in_dim: input size.
#     h_dim: Dimension of hidden layer.
#     out_dim: Output dimension.
#     num_layers: Number of conv layers.
#     dropout: Dropout probability for hidden layers.
#     model: "rsage" or "rgat".
#     heads: Number of multi-head-attentions for GAT.
#     node_type: The predict node type for node classification.

#   """
    def __init__(self, etypes, in_dim, h_dim, out_dim, num_layers=2, dropout=0.2, model='rgat', heads=4, node_type=None, with_trim=False):
        super().__init__()
        self.node_type = node_type
        if node_type is not None:
            self.lin = torch.nn.Linear(h_dim, out_dim)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_dim if i == 0 else h_dim
            h_dim = out_dim if (i == (num_layers - 1) and node_type is None) else h_dim
            if model == 'rsage':
                self.convs.append(HeteroGraphConv({
                    etype: SAGEConv(in_dim, h_dim, root_weight=False)
                    for etype in etypes}))
            elif model == 'rgat':
                self.convs.append(HeteroGraphConv({
                    etype: GATConv(in_dim, h_dim // heads, heads)
                    for etype in etypes}))
        self.dropout = torch.nn.Dropout(dropout)
        self.with_trim = with_trim

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.convs, blocks)):
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            if l != len(self.convs) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        return self.lin(h['paper'])   