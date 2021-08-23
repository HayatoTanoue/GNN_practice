from torch_geometric.utils import from_networkx, to_networkx
from glob import glob
from tqdm import tqdm

import torch
import networkx as nx


def make_datasets(x="degree"):

    data_list = []
    for i, file in enumerate(sorted(glob("/workspace/my_data/*"))):
        # adjlist のパスを取得
        paths = glob(file + "/*.adjlist")
        data_list += [data_from_adjlist(p, i, x) for p in tqdm(paths)]
    return data_list


# adjlist -> data (torch geometric)
def data_from_adjlist(path, label, x="degree"):
    """adjlist to torch geometric data"""
    # load adjlist
    G = nx.read_adjlist(path)
    # graph to torch geometric data
    data = from_networkx(G)

    # set node attribute
    if x == "degree":
        data.x = torch.tensor(
            [[d] for d in list(dict(G.degree()).values())],
            dtype=torch.float,
        )
    else:
        data.x = torch.tensor([[float(1)] for i in range(nx.number_of_nodes(G))])

    # set graph label
    data.y = torch.tensor([label])
    return data


# cv_train
def cv_train(model, loader, device, criterion, optimizer, model_name="GCN"):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.

        data.to(device)
        if model_name == "GCN":
            out = model(
                data.x, data.edge_index, data.batch
            )  # Perform a single forward pass.
        elif model_name == "DGCNN":
            out = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


# cv_valid
def cv_test(model, loader, device, model_name="GCN"):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        if model_name == "GCN":
            out = model(
                data.x, data.edge_index, data.batch
            )  # Perform a single forward pass.
        elif model_name == "DGCNN":
            out = model(data)  # Perform a single forward pass.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.

    val_acc = correct / len(loader.dataset)  # Derive ratio of correct predictions.
    return val_acc
