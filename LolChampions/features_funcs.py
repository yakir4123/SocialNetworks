import statistics
import numpy as np
import networkx as nx


class FeatureExtractor:

    def __init__(self, G: nx.Graph = None):
        self.G = G
        self.paths = {}
        self.weighted_centrality = {}
        self.unweighted_centrality = {}
        self.debug = 0

    def setG(self, G: nx.Graph):
        self.G = G
        self.initial_graph_features()

    def node_weight(self, u):
        """
        Get the weight of all edges connected to u
        """

        return sum(self.G[u][v]['weight'] for _, v in self.G.edges(u))

    def edge_weighted_betweenness_centrality(self, u, v):
        return self.G[u][v]['weighted_betweenness']

    def edge_unweighted_betweenness_centrality(self, u, v):
        return self.G[u][v]['unweighted_betweenness']

    def var_k_path_weight(self, u, v, k=2):
        """
        get std of all paths between u and v n nodes
        """
        e = edge_as_key(u, v)
        if e not in self.paths.keys():
            # get all paths with n - 1 edges
            self.paths[e] = list(nx.all_simple_paths(self.G, e[0], e[1], cutoff=k - 1))

        try:
            l = [self.G[u2][v2]['weight']
                                      for path in self.paths[e]  # iterate over all paths from u to v
                                      for u2, v2 in zip(path, path[1:])  # iterate over all edges from path
                                      if len(path) == k]
            self.debug += len(self.paths[e])
            var = statistics.variance(l)  # choose only those with len == k
            return var
        except statistics.StatisticsError as e:
            # no paths exist
            return 0

    def avarage_k_path_weight(self, u, v, k=2):
        """
        get average of all paths between u and v with k nodes
        """
        e = edge_as_key(u, v)
        if e not in self.paths:
            # get all paths with n - 1 edges
            self.paths[e] = list(nx.all_simple_paths(self.G, e[0], e[1], cutoff=k - 1))
        try:
            return statistics.mean(self.G[u2][v2]['weight'] for path in self.paths[e]
                                   for u2, v2 in zip(path, path[1:])
                                   if len(path) == k)
        except statistics.StatisticsError as e:
            return 0

    def sign(self, u, v):
        return np.sign(self.G[u][v]['weight'])

    def initial_graph_features(self):
        """
        Calculate betweenness on graph weighted and unweighted
        """
        bb = nx.edge_betweenness_centrality(self.G, normalized=True)
        nx.set_edge_attributes(self.G, bb, 'unweighted_betweenness')
        bb = nx.edge_betweenness_centrality(self.G, normalized=True, weight='weight')
        nx.set_edge_attributes(self.G, bb, 'weighted_betweenness')

    def score(self, u, v):
        return self.G[u][v]['weight'] / self.G[u][v]['matches']
        pass

    def lane(self, u):
        return self.G.nodes[u]['role']

    def role(self, u):
        return self.G.nodes[u]['lane']


def edge_as_key(u, v):
    return (u, v) if u < v else (v, u)
