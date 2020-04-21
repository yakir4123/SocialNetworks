import pandas as pd
from itertools import combinations


class Graph:

    def __init__(self):

        self.graph = {}                 # {source:[dest]}
        self.rev_graph = {}             # {dest:[source]}
        self.PR = {}                    # {node: [pr[t=0], pr[t=1],..]}
        self.panda_graph = None         # pandas DataFrame [No, Node, Neighbors, Degree, PageRank, CC(Undirected)]
        self.CC = {}                    # {node: cc}
        self.degrees = {}               # {node : degree}
        self.nodes = []                 # [node_1, node_2,..]
    
    def load_graph(self, path):

        df = pd.read_csv(path, names=['source', 'dest'])
        self.graph = df.applymap(int).groupby('source')['dest'].apply(list).to_dict()
        self.rev_graph = df.applymap(int).groupby('dest')['source'].apply(list).to_dict()
        self.panda_graph = pd.DataFrame(self.graph.items(), columns=['Node', 'Neighbors']).sort_values(by=['Node'])
        self.panda_graph["Degree"] = list(map(lambda node: len(self.graph[node]), self.graph.keys()))
        self.degrees = dict(zip(self.panda_graph.Node, self.panda_graph.Degree))
        self.nodes = list(self.graph.keys())

    def calculate_page_rank(self, β=0.85, δ=0.001, max_iterations=20):

        all_nodes = list(set(self.nodes) - set(self.rev_graph))
        N = len(all_nodes + self.nodes)

        for node in self.nodes:
            self.PR[node] = [1 / N]
        t = 0

        def division(dest, t):
            try:
                return self.PR[dest][t - 1] / self.degrees[dest]
            except:
                return 0

        while True:

            t += 1

            for node in self.nodes:
                try:
                    self.PR[node].append(β * sum(map(lambda dest: division(dest, t), self.rev_graph[node])))
                except:
                    self.PR[node].append(0)

            S = (1 - sum(map(lambda node: self.PR[node][t], self.nodes)))/N

            for node in self.nodes:
                self.PR[node][t] += S

            if t == max_iterations + 1 or sum(map(lambda n: abs(self.PR[n][t] - self.PR[n][t - 1]), self.nodes)) <= δ:
                self.panda_graph["PageRank"] = list(map(lambda n: self.PR[n][-1], self.nodes))
                print("PageRank func stop at iteration {iter}".format(iter=t))
                break

    def get_page_rank(self, node_name):

        if node_name not in self.PR:
            return -1
        return self.PR[node_name][-1]

    def get_top_page_rank(self, n=1):
        return self.get_all_page_rank()[:n]

    def get_all_page_rank(self):

        rank = list(map(lambda node: (node, self.PR[node][-1]), self.nodes))
        rank.sort(key=lambda x: x[1], reverse=True)
        return rank

    def calculate_cc(self):

        def is_neighbors(node_1, node_2):
            try:
                if node_1 in self.graph[node_2] or node_2 in self.graph[node_1]:
                    return 1
                return 0
            except:
                return 0

        for node in self.nodes:
            k = self.degrees[node]
            e = sum(map(lambda pair: is_neighbors(*pair), list(combinations(self.graph[node], 2))))
            if k <= 1:
                self.CC[node] = 0
            else:
                self.CC[node] = 2 * e / (k * (k-1))

        self.panda_graph["CC (Undirected)"] = self.CC.values()

    def get_cc(self, node_name):

        if node_name not in self.CC:
            return -1
        return self.CC[node_name]

    def get_top_cc(self, n):
        return self.get_all_cc()[:n]

    def get_all_cc(self):

        cc = list(map(lambda node: (node, self.CC[node][-1]), self.nodes))
        cc.sort(key=lambda x: x[1], reverse=True)
        return cc

    def get_average_cc(self):

        vals = self.CC.values()
        return sum(vals)/len(vals)



G = Graph()
G.load_graph("Wikipedia_votes.csv")
G.calculate_page_rank()
G.calculate_cc()
print("Top PageRank: {ranks}".format(ranks = G.get_top_page_rank(5)))
print("Top CC: {cc}".format(cc = G.get_top_cc(5)))
print("Average CC: {avg}".format(avg =G.get_average_cc()))
print(G.panda_graph)




