import os
import torch
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from glob import glob
from sklearn.metrics import confusion_matrix
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.data import DataLoader


def data_from_adjlist(path, label, x=None):
    G = nx.read_adjlist(path)
    data = from_networkx(G)
    if x == "degree_center":
        data.x = torch.tensor(
            [[d] for d in list(dict(nx.degree_centrality(G)).values())],
            dtype=torch.float,
        )
    elif x == "degree":
        data.x = torch.tensor([[float(d)] for d in list(dict(G.degree()).values())])
    else:
        data.x = torch.tensor([[float(1)] for i in range(nx.number_of_nodes(G))])
    data.y = torch.tensor([label])
    return data


def make_datasets(batch_size=32, split_rate=0.7):
    train_data_list = []
    test_data_list = []
    for i, file in enumerate(sorted(glob("/workspace/my_data/*"))):
        # adjlist のパスを取得
        paths = glob(file + "/*.adjlist")

        index = int(len(paths) * split_rate)
        train_data_list += [data_from_adjlist(p, i) for p in paths[:index]]
        test_data_list += [data_from_adjlist(p, i) for p in paths[index:]]

    train_data_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_loader


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


def conf_matrix(model, data_loader, device, benchmark=False, benchmark_class=0, x=None):
    """混同行列の作成"""
    y_pred = []
    y_true = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            data = make_x(data, x)
            data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            y_pred += pred.cpu()
            y_true += data.y.cpu()

    # print(y_pred[:5])
    # print(y_true[:5])

    # 各クラスの名前抽出
    if benchmark:
        from collections import Counter

        classes = [str(n) for n in range(benchmark_class)]
    else:
        try:
            classes = [
                i.split("_")[1] for i in sorted(os.listdir("/workspace/my_data/"))
            ]
        except:
            classes = [i for i in sorted(os.listdir("/workspace/my_data/"))]
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    # print(classes)
    # print(cf_matrix)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix),
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    # plt.figure(figsize=(12, 7))
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    sns.set_style()
    sns.set(font_scale=1.7)
    sns.heatmap(df_cm, annot=True, ax=ax)
    plt.tight_layout()

    return fig, ax
