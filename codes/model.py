from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import torch


class GCN(torch.nn.Module):
    def __init__(
        self, hidden_channels, dataset=None, num_classes=4, num_node_feature=1
    ):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        if dataset is not None:
            if dataset.num_node_features == 0:
                self.conv1 = GCNConv(1, hidden_channels)
            else:
                self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        else:
            self.conv1 = GCNConv(num_node_feature, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # self.conv1 = GATConv(dataset.num_node_features, hidden_channels)
        # self.conv2 = GATConv(hidden_channels, hidden_channels)
        # self.conv3 = GATConv(hidden_channels, hidden_channels)
        if dataset is not None:
            self.lin = Linear(hidden_channels, dataset.num_classes)
        else:
            self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
