import pandas as pd


class Graph:

    def __init__(self):

        self.graph = {} # {source:[dest]}
        self.reverse_graph = {} # {dest:[source]}

    def load_graph(self, path):

        df = pd.read_csv(path, names=['source', 'dest'])
        self.graph = df.applymap(int).groupby('source')['dest'].apply(
            list).to_dict()
        self.reverse_graph = df.applymap(int).groupby('dest')['source'].apply(
            list).to_dict()

    def get_nodes(self):
        return list(self.graph.keys())

    def degree(self, node):
        return len(self.graph[node])

    def calculate_page_rank(self, β = 0.85, δ = 0.001, maxIterations = 20):

        nodes = self.get_nodes()
        N = len(self.get_nodes())
        PR = dict(zip(nodes, [[1/N]] * N))
        t = 1

        def devision(dest, t):
            return PR[dest][t - 1] / self.degree(dest)

        while(True):

            for node in nodes:
                if self.degree(node) == 0:
                    PR[node].append(0)
                else:
                    print(self.graph[node])
                    PR[node].append(β * sum(lambda dest: devision(dest,t), self.graph[node]))

            S = (1 + sum(lambda node: PR[node][t], nodes))/N

            for node in nodes:
                PR[node][t] = (1 - S)/N

            t += 1


G = Graph()
G.graph = {'A':['B','C','D'], 'B':['D','A'], 'C':['D'], 'D':['B','C']}
G.calculate_page_rank()
G.load_graph("Wikipedia_votes.csv")
print(G.graph)
