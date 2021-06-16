#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

便利な関数ファイル

優先的選択なし : no_preferential_attachment(n, m)
成長なし : no_growth_barabasi(n, step)

次数分布 : degree_hist(G, ax, log=False, bins=50)
ネットワークを画像に変換 : network_to_image(G)

"""

import networkx as nx
import numpy as np
import math
from PIL import Image


def no_preferential_attachment(n, m):
    """
    優先的選択がないバラバシアルバートグラフ

    network science chapter5.6 model A

    Parameters
    ----------
    n : int
        number of nodes.
    m : int
        number of add node edge.

    Returns
    -------
    G : networkx graph

    """

    G = nx.empty_graph(m)  # generate m nodes empty graph

    for i in range(n - m):
        nodes = list(G.nodes())
        new_node = nodes[-1] + 1
        G.add_node(new_node)

        selected = np.random.choice(nodes, size=m, replace=False)

        G.add_edges_from([(new_node, select) for select in selected])

    return G


def no_growth_barabasi(n, step):
    """
    成長のないバラバシアルバートグラフ

    network science chapter5.6 model B

    Parameters
    ----------
    n : int
        number of nodes.
    step : int
        number of steps.

    Returns
    -------
    G : networkx graph
    """
    G = nx.empty_graph(n)  # generate n nodes empty graphe

    def select_by_degree(G):
        """次数に比例した選択確率によるノード抽選"""
        degree = [G.degree(i) + 1 for i in range(n)]  # 次数0 でも抽選されるように+1を行う
        total_degree = sum(degree)
        weight = [i / total_degree for i in degree]
        selected = np.random.choice(range(n), p=weight)

        return selected

    s = 0
    while s < step:
        # 各ステップで無作為に抽選したノードからエッジを一つ張る
        now = np.random.randint(n)

        selected = select_by_degree(G)
        while now == selected:
            selected = select_by_degree(G)

        G.add_edge(now, selected)

        s += 1

    return G


def degree_hist(G, ax, log=False, bins=50, alpha=1, color="r", label=None):
    """次数分布を作成

    Parameter
    ----------
    G : networkx graph
    ax : matplotlib axes (グラフの描画先)
    log : bool
        対数グラフにするか (default = False)
    bins : int
        ヒストグラムの階級数
    """

    # 次数リストの作成
    degs = [d for d in dict(G.degree()).values()]

    # 対数
    if log:
        # データ範囲の最大値 = log10(ノード数)
        num = math.log10(nx.number_of_nodes(G))
        ax.hist(
            degs,
            bins=np.logspace(0, num, bins),
            density=True,
            alpha=alpha,
            color=color,
            label=label,
        )
        ax.set_xscale("log")  # x scale to log
        ax.set_yscale("log")  # y scale to log
    # 線形
    else:
        ax.hist(degs, bins, density=True, alpha=alpha, color=color, label=label)


def network_to_image(G, sort=False):
    """ネットワークを画像データに変換する

    ネットワークの隣接行列を作成し、
    行列に255を乗算する (エッジの有無が画像の白黒によって表現される)
    行列を画像に変換する(グレースケール)

    Parameters
    ----------
    G : networkx.graph.Graph
        単純グラフ
    Returns
    -------
    PIL.Image
        ネットワークの隣接行列を画像化したもの
    """

    def _sort_by_degree(A, G):
        # 隣接行列を次数の昇順に並び替える

        # 次数の辞書を取得
        degs = dict(G.degree())
        # value(次数)で並び替え
        sort_degs = sorted(degs.items(), key=lambda x: x[1])
        sort_nodes = [node[0] for node in sort_degs]

        # 行, 列並び替え
        A = A[:, sort_nodes]
        A = A[sort_nodes, :]
        return A

    assert type(G) == nx.Graph, "input graph must be networkx.Graph"

    # 隣接行列の作成
    A = nx.to_numpy_array(G)

    if sort:
        A = _sort_by_degree(A, G)

    # array to image
    img = Image.fromarray(A * 255).convert("L")

    return img


def image_to_network(path):
    """画像を読み込んでネットワークにする

    Parameters
    ----------
    path : str
        path to image

    Returns
    -------
    G : networkx graph

    """
    img = Image.open(path).convert("L")
    array = np.array(img)

    G = nx.from_numpy_array(array)

    return G
