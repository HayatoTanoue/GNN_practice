import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx


def make_x(data, x=None):
    if data.x is None:
        G = to_networkx(data)
        if x == "degree":
            data.x = torch.tensor([[float(d)] for d in list(dict(G.degree()).values())])
        else:
            data.x = torch.tensor(
                np.array([[1] for d in range(nx.number_of_nodes(G))])
            ).float()
    return data


def train(model, train_loader, criterion, optimizer, device, x=None):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        data = make_x(data, x)
        data.to(device)
        out = model(
            data.x, data.edge_index, data.batch
        )  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(model, loader, device, x=None):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = make_x(data, x)
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.
