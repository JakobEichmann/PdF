import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class GNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data: Data, return_node_embeddings: bool = False):
        # добавим фиктивный batch=0
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        graph_emb = global_mean_pool(x, batch)  # [1, out_channels]
        if return_node_embeddings:
            return graph_emb, x  # x: [num_nodes, out_channels]
        return graph_emb


def encode_graph(data: Data) -> torch.Tensor:
    """Старый интерфейс – только графовый embedding (для совместимости)."""
    model = GNNEncoder(in_channels=data.num_node_features)
    model.eval()
    with torch.no_grad():
        emb = model(data, return_node_embeddings=False)
    return emb.squeeze(0)


def encode_graph_with_nodes(data: Data):
    """
    Новый интерфейс: возвращает
    - graph_emb: [out_channels]
    - node_embs: [num_nodes, out_channels]
    """
    model = GNNEncoder(in_channels=data.num_node_features)
    model.eval()
    with torch.no_grad():
        graph_emb, node_embs = model(data, return_node_embeddings=True)
    return graph_emb.squeeze(0), node_embs


if __name__ == "__main__":
    # локальный тест – оставляем
    x = torch.eye(4)
    edge_index = torch.tensor([[0, 1, 2, 3],
                               [1, 2, 3, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    emb, node_embs = encode_graph_with_nodes(data)
    print("graph embedding shape:", emb.shape)
    print("node embeddings shape:", node_embs.shape)
