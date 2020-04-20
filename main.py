import pandas as pd
from itertools import combinations

class Graph:

    def __init__(self):

        self.graph = {} # {source:[dest]}
        self.PR = {}
        self.panda_graph = None
        self.CC = {}

    def load_graph(self, path):

        df = pd.read_csv(path, names=['source', 'dest'])
        self.graph = df.applymap(int).groupby('source')['dest'].apply(list).to_dict()
        self.panda_graph = pd.DataFrame(self.graph.items(), columns=['Node', 'Neighbors']).sort_values(by=['Node'])

    def get_nodes(self):
        return list(self.graph.keys())

    def degree(self, node):
        return len(self.graph[node])

    def calculate_page_rank(self, β=0.85, δ=0.001, maxIterations=20):

        nodes = self.get_nodes()
        N = len(self.get_nodes())
        for node in nodes:
            self.PR[node] = [1 / N]
        t = 0

        def devision(dest, t):

            try:
                return self.PR[dest][t - 1] / self.degree(dest)
            except:
                return 0

        while True:

            t += 1

            for node in nodes:
                if node == 214:
                    pass
                if node not in self.graph or self.degree(node) == 0:
                    self.PR[node].append(0)
                else:
                    self.PR[node].append(β * sum(map(lambda dest: devision(dest,t), self.graph[node])))

            S = (1 - sum(map(lambda node: self.PR[node][t], nodes)))/N

            for node in nodes:
                self.PR[node][t] += S

            if t == maxIterations + 1 or sum(map(lambda n: abs(self.PR[n][t] - self.PR[n][t - 1]), nodes)) <= δ:
                self.panda_graph["PageRank"] = list(map(lambda n: self.PR[n][-1], nodes))
                break

        # for node in nodes:
        #     print("node {key} : PageRank {val}".format(key=node, val=self.PR[node][-1]))

    def get_PageRank(self, node_name):

        if node_name not in self.PR:
            return -1
        return self.PR[node_name][-1]

    def get_top_PageRank(self, n=1):
        return self.panda_graph.nlargest(n, "PageRank")

    def get_all_PageRank(self):

        rank = list(map(lambda raw: (raw[1],raw[3]), self.panda_graph.to_records()))
        rank.sort(key=lambda x: x[1], reverse=True)
        return rank

    def calculate_CC(self):

        nodes = self.get_nodes()

        def is_neighbors(node_1, node_2):
            try:
                if node_1 in self.graph[node_2] or node_2 in self.graph[node_1]:
                    return 1
                else:
                    return 0
            except:
                return 0

        for node in nodes:
            k = self.degree(node)
            e = sum(map(lambda pair: is_neighbors(*pair), list(combinations(self.graph[node], 2))))
            if k <= 1:
                self.CC[node] = 0
            else:
                self.CC[node] = 2 * e / (k * (k-1))

        self.panda_graph["CC (Undirected)"] = self.CC.values()

G = Graph()
G.load_graph("Wikipedia_votes.csv")
G.calculate_page_rank()
G.calculate_CC()
print(G.panda_graph)
