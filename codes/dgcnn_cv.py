import torch
import networkx as nx
import numpy as np


from glob import glob
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.data import DataLoader

# from model import GCN

from DGCNN import DGCNN_Model


class Solver(object):
    def __init__(
        self,
        n_splits=10,
        random_state=0,
        num_epochs=10,
        num_batchs=100,
        hidden_channels=64,
        num_classes=4,
        lr=0.01,
    ):
        super().__init__()
        self.n_splits = n_splits
        self.random_state = random_state
        self.num_epochs = num_epochs
        self.num_batchs = num_batchs
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.num_node_feature = 1
        self.dataset = None
        self.lr = lr

    def make_datasets(self):
        data_list = []
        for i, file in enumerate(sorted(glob("/workspace/my_data/*"))):
            # adjlist のパスを取得
            paths = glob(file + "/*.adjlist")
            data_list += [self.data_from_adjlist(p, i) for p in tqdm(paths)]

        self.data_list = data_list

    # adjlist -> data (torch geometric)
    def data_from_adjlist(self, path, label, x="degree_center"):
        """adjlist to torch geometric data"""
        # load adjlist
        G = nx.read_adjlist(path)
        # graph to torch geometric data
        data = from_networkx(G)

        # set node attribute
        if x == "degree_center":
            data.x = torch.tensor(
                [[d] for d in list(dict(G.degree()).values())],
                dtype=torch.float,
            )
        else:
            data.x = torch.tensor([[float(1)] for i in range(nx.number_of_nodes(G))])

        # set graph label
        data.y = torch.tensor([label])
        return data

    def setModel(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model = GCN(
        #     self.hidden_channels, self.dataset, self.num_classes, self.num_node_feature
        # )

        self.model = DGCNN_Model(self.num_node_feature, self.num_classes)
        self.model.to(self.device)
        # self.optimizer = torch.optim.Adagrad(
        #     self.model.parameters(), lr=self.lr, eps=1e-5
        # )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.criterion = torch.nn.CrossEntropyLoss()

    # cv_train
    def cv_train(self, loader):
        self.model.train()

        for data in loader:  # Iterate in batches over the training dataset.
            data.to(self.device)
            out = self.model(data)  # Perform a single forward pass.
            loss = self.criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.

    # cv_valid
    def cv_test(self, loader):
        self.model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.to(self.device)
            out = self.model(data)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.

        val_acc = correct / len(loader.dataset)  # Derive ratio of correct predictions.
        return val_acc

    # Cross_Valid_Train
    def Cross_Valid_Train(self):
        """交差検証"""
        # cv = 0.0
        fold = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        valid_accs = []
        for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(self.data_list)):

            print("fold {}".format(fold_idx))

            self.setModel()

            train_loader = DataLoader(
                Subset(self.data_list, train_idx),
                shuffle=True,
                batch_size=self.num_batchs,
            )
            valid_loader = DataLoader(
                Subset(self.data_list, valid_idx),
                shuffle=False,
                batch_size=self.num_batchs,
            )

            for epoch_idx in range(self.num_epochs):

                self.cv_train(train_loader)
                valid_acc = self.cv_test(valid_loader)

                print(valid_acc)

            valid_accs.append(valid_acc)

        return valid_accs
