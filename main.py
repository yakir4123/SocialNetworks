import pandas as pd

class Graph:

    def __init__(self):

        self.graph = {} # {source:[dest]}
        self.PR = {}

    def load_graph(self, path):

        df = pd.read_csv(path, names=['source', 'dest'])
        self.graph = df.applymap(int).groupby('source')['dest'].apply(
            list).to_dict()

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
                break

        for node in nodes:
            print("node {key} : PageRank {val}".format(key=node, val=self.PR[node][-1]))

G = Graph()
G.load_graph("Wikipedia_votes.csv")
G.calculate_page_rank()
print(G.graph)
