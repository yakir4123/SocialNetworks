import numpy as np
import networkx as nx


def node_weight(G: nx.Graph, u):
    """
    Get the weight of all edges connected to u
    """

    return sum(G[u][v]['weight'] for _, v in G.edges(u))


def std_k_path_weight(G: nx.Graph, u, v, k=2):
    """
    get std of all paths between u and v n nodes
    """
    # get all paths with n - 1 edges
    paths = nx.all_simple_paths(G, u, v, cutoff=k - 1)
    res = 0
    n = 0
    for path in paths:
        if len(path) == k:
            n += 1
            res += sum(G[u2][v2]['weight'] for u2, v2 in zip(path, path[1:]))
    return 0 if n == 0 else res / (n * (k - 1))


def avarage_k_path_weight(G: nx.Graph, u, v, k=2):
    """
    get average of all paths between u and v with n nodes
    """
    # get all paths with n - 1 edges
    paths = nx.all_simple_paths(G, u, v, cutoff=k - 1)
    res = 0
    n = 0
    for path in paths:
        if len(path) == k:
            n += 1
            res += sum(G[u2][v2]['weight'] for u2, v2 in zip(path, path[1:]))
    return 0 if n == 0 else res / (n * (k - 1))


def sign(G: nx.Graph, u, v):
    return np.sign(G[u][v]['weight'])
