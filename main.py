import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

class Graph:

    def __init__(self):

        self.graph = {}                 # {source:[dest]}
        self.rev_graph = {}             # {dest:[source]}
        self.PR = {}                    # {node: [pr[t=0], pr[t=1],..]}
        self.panda_graph = None         # pandas DataFrame [No, Node, Neighbors, Degree, PageRank, CC(Undirected)]
        self.CC = {}                    # {node: cc}
        self.degrees = {}               # {node : degree}
        self.nodes = []                 # [node_1, node_2,..]
        self.edges = []                 # [(neighbors_1, neighbor_2)]

    def load_graph(self, path):
        """
        Loads a graph from a text file to the memory.
        The csv file is given as an edge list – i.e, <source, destination> pairs of node names.
        :param path: String, the path for the input file
        :return: None
        """

        df = pd.read_csv(path, names=['source', 'dest'])
        self.graph = df.applymap(int).groupby('source')['dest'].apply(list).to_dict()
        self.rev_graph = df.applymap(int).groupby('dest')['source'].apply(list).to_dict()
        self.panda_graph = pd.DataFrame(self.graph.items(), columns=['Node', 'Neighbors']).sort_values(by=['Node'])
        self.panda_graph["Degree"] = list(map(lambda node: len(self.graph[node]), self.graph.keys()))
        self.degrees = dict(zip(self.panda_graph.Node, self.panda_graph.Degree))
        self.nodes = list(self.graph.keys())

    def calculate_page_rank(self, β=0.85, δ=0.001, max_iterations=20):
        """
        Calculates the PageRank for each of the nodes in the graph.
        The calculation will end in one of two stop conditions: after maxIterations iterations,
        or when the difference between two iterations is smaller than δ.
        Save the results into an internal data structure for an easy access
        :param β: Double, defines how important the centrality of
        :param δ: Minimal PageRank change between iterations
        :param max_iterations: Int, maximum number of page rank iterations
        :return: None
        """

        all_nodes = list(set(self.graph.keys()).union(set(self.rev_graph.keys())))
        # num of nodes
        N = len(all_nodes)
        print("Num of nodes: " + str(N))

        # reset all nodes PageRank to 1/N at time t=0
        for node in all_nodes:  # 𝑟_𝑗(0) = 0
            self.PR[node] = [1 / N]
        t = 0

        def division(dest, t): # return 𝑟′𝑖(𝑡)/𝑑_𝑖
            try:
                return self.PR[dest][t - 1] / self.degrees[dest]
            except:
                return 0

        while True:
            t += 1
            for node in all_nodes:  # 𝑟′𝑗(𝑡)_ = sum 𝑖->𝑗: (β * 𝑟′𝑖(𝑡)/𝑑_𝑖)
                try:
                    self.PR[node].append(β * sum(map(lambda dest: division(dest, t), self.rev_graph[node])))
                except:
                    self.PR[node].append(0)

            # (1 - sum 𝑗: 𝑟′𝑗(𝑡))/N
            S = (1 - sum(map(lambda node: self.PR[node][-1], all_nodes)))/N

            for node in all_nodes:
                self.PR[node][-1] += S  # 𝑟_𝑗(𝑡) = 𝑟′𝑗(𝑡) + (1 - sum 𝑗: 𝑟′𝑗(𝑡))/N

            # while sum j: 𝑟_𝑗(𝑡) - 𝑟_𝑗(𝑡 - 1) > δ or t = max_iterations
            if t == max_iterations or sum(map(lambda n: abs(self.PR[n][-1] - self.PR[n][-2]), all_nodes)) <= δ:
                self.panda_graph["PageRank"] = list(map(lambda n: self.PR[n][-1], self.nodes))
                print("PageRank func stop at iteration {iter}".format(iter=t))
                break

    def get_page_rank(self, node_name):
        """
        Returns the PageRank of a specific node.
        Return “-1” for non-existing name
        :param node_name: String, The node name
        :return: Double:, The PageRank of the given node name
        """
        if node_name not in self.PR:
            return -1
        return self.PR[node_name][-1]

    def get_top_page_rank(self, n=1):
        """
        Returns a list of n nodes with the highest PageRank value.
        The ordered according to the PageRank values from high to low
        :param n: Integer, How many nodes
        :return: List of pairs: <node name, PageRank value >
        """
        return self.get_all_page_rank()[:n]

    def get_all_page_rank(self):
        """
        Returns a list of the PageRank for all the nodes in the graph
        The list should include pairs of node id, PageRank value of that node.
        The list should be ordered according to the PageRank values from high to low
        :return: List of pairs: <node name, PageRank value >
        """
        if self.PR == {}:
            return -1
        rank = list(map(lambda node: (node, self.PR[node][-1]), self.nodes))
        rank.sort(key=lambda x: x[1], reverse=True)
        return rank

    def calculate_cc(self):
        """
        Calculates the (undirected) clustering coefficient of each of the nodes in the graph
        :return: None
        """
        def is_neighbors(node_1, node_2):
            """
            check if node_1 and node_2 are neighbors
            :param node_1: int, first node
            :param node_2: int, second node
            :return: 1 if node_1 and node_2 are neighbors, 0 else
            """
            try:
                if node_1 in self.graph[node_2] or node_2 in self.graph[node_1]:
                    return 1
                return 0
            except:
                return 0

        for node in self.nodes: # 𝐶𝑖 = 𝑒_𝑖 / (|Γ_𝑖_|(|Γ_𝑖_|−1))
            k = self.degrees[node]
            e = sum(map(lambda neighbor: is_neighbors(node, neighbor), self.graph[node]))

            if k <= 1:
                self.CC[node] = 0
            else:
                self.CC[node] = e / (k * (k-1))

        self.panda_graph["CC (directed)"] = self.CC.values()

    def get_cc(self, node_name):
        """
        Returns the clustering coefficient of a specific node.
        Return “-1” for non-existing name
        :param node_name: String, The node name
        :return: Double, The CC of the given node name
        """
        if node_name not in self.CC:
            return -1
        return self.CC[node_name]

    def get_top_cc(self, n):
        """
        Returns a list of n nodes with the highest clustering coefficients.
        The list should be ordered according to the CC values from high to low
        :param n: Integer, How many nodes
        :return: List of pairs: <node name, CC value >
        """
        if self.CC == {}:
            return -1
        return self.get_all_cc()[:n]

    def get_all_cc(self):
        """
        Returns a list of the clustering coefficients for all the nodes in the graph
        The list should include pairs of node id, CC value of that node.
        The list should be ordered according to the CC values from high to low
        :return: List of pairs: <node name, CC value >
        """
        if self.CC == {}:
            return -1
        cc = list(map(lambda node: (node, self.CC[node]), self.nodes))
        cc.sort(key=lambda x: x[1], reverse=True)
        return cc

    def get_average_cc(self):
        """
        Returns a simple average over all the clustering coefficients values
        :return: Double, average CC
        """
        if self.CC == {}:
            return -1
        vals = self.CC.values()
        return sum(vals)/len(vals)

    def plot(self):
        for node in self.nodes:
            self.edges += list(map(lambda neighbor: (node, neighbor), self.graph[node]))
        print(self.edges)
        graph_plot = nx.Graph()
        graph_plot.add_nodes_from(self.edges)
        compression_opts = dict(method='zip',archive_name='got-result.csv')
        self.panda_graph.to_csv('got-result.zip', index=False,compression=compression_opts)
        nx.draw(graph_plot)
        plt.show()

# G = Graph()
# G.load_graph("GOT.csv")
# G.calculate_page_rank()
# G.calculate_cc()
# print("Top PageRank: {ranks}".format(ranks = G.get_top_page_rank(10)))
# print("Top CC: {cc}".format(cc = G.get_top_cc(10)))
# print("Average CC: {avg}".format(avg =G.get_average_cc()))
# print(G.panda_graph)
# G.plot()




