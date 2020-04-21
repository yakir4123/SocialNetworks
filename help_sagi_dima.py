# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:30:46 2020
@author: Dima & Sagi
"""

import pandas as pd
import numpy as np
import operator
import heapq
from statistics import mean


class graph:

    def __init__(self):

        self.source_to_dest_dict = {}
        self.dest_to_source_dict = {}
        self.page_rank = {}
        self.cc = {}

    def load_graph(self, path):
        """
        The function loads the csv file(graph)
        :param path: path of the csv file
        :return: None
        """
        df = pd.read_csv(path, names=['source', 'dest'])
        self.source_to_dest_dict = df.applymap(str).groupby('source')['dest'].apply(
            list).to_dict()  # src points to dest
        self.dest_to_source_dict = df.applymap(str).groupby('dest')['source'].apply(
            list).to_dict()  # dest who are pointing at the source

    def calculate_page_rank(self, beta=0.85, delta=0.001, maxIterations=20):
        """
        Calculates the PageRank for each of the nodes in the graph.
        The calculation will end in one of two stop conditions:
        after maxIterations iterations, or when the difference
        between two iterations is smaller than δ.

        :param beta:
        :param delta:
        :param maxIterations:
        :return: None
        """
        # s2d = {A: [B, C], B:[D], C: [A, B, D], D: [C]}
        # d2s = {A: [C], B: [A, C], C: [A, D], D: [B, C]}
        s2d = self.source_to_dest_dict
        d2s = self.dest_to_source_dict

        N = len(s2d.keys())

        max1 = max(s2d.keys(), key=lambda x: int(x))
        max2 = max(d2s.keys(), key=lambda x: int(x))
        N = int(max(max1, max2))

        if N == 0 or N == 1:
            print("not a valid node realation to run Pagerank")
            return

        pagerank = np.full((N, maxIterations), 0, dtype="float")
        for i in range(N):
            pagerank[i][0] = 1 / N
        temp_pagerank = np.full((N, maxIterations), 0, dtype="float")
        time = range(1, maxIterations)
        pr_sum = [0] * (len(time) + 1)
        pr_sum[0] = N
        s = 0

        # The Pagerank algorithm
        for t in time:
            for temp_node in d2s.keys():
                for pr_node in d2s[temp_node]:
                    # r'_j(t)= beta*r_i(t-1)/d_i
                    temp_pagerank[int(temp_node) - 1][t] += beta * pagerank[int(pr_node) - 1][t - 1] / len(s2d[pr_node])

                s += temp_pagerank[int(temp_node) - 1][t]
                if temp_node == list(d2s.keys())[-1]:
                    pass
            for temp_node in d2s.keys():
                pagerank[int(temp_node) - 1][t] = temp_pagerank[int(temp_node) - 1][t] + (1 - s) / N
                pr_sum[t] += pagerank[int(temp_node) - 1][t]
            if abs(pr_sum[t] - pr_sum[t - 1]) < delta:
                print("We have reached less then delta = ", delta, " and after ", t,
                      " iterations.\n Pagerank scores achieved.")
                for i in range(N):
                    self.page_rank.update({str(i + 1): pagerank[i][t]})
                break
            s = 0

        print("Pagerank algorithm is done")

    def get_PageRank(self, node_name):
        """
        Returns the PageRank of a specific node.
        Return “-1” for non-existing name
        :param node_name: The node name
        :return: The PageRank of the given node name
        :rtype: double
        """
        if node_name in self.page_rank:
            return self.page_rank[node_name]
        else:
            return -1

    def get_top_PageRank(self, n):
        """
        Returns a list of n nodes with the highest PageRank value.
        The list should be ordered according to the PageRank values from high to low
        :params n: How many nodes.
        :return: List of pairs:<node name, PageRank value >
        :rtype: list.
        """
        return heapq.nlargest(n, self.page_rank.items(), key=lambda i: i[1])

    def get_all_PageRank(self):
        """
        Returns a list of the PageRank for all the nodes in the graph
        The list should include pairs of node id, PageRank value of that node.
        The list should be ordered according to the PageRank values from high to low
        :params: None
        :return: List of pairs:<node name, PageRank value >
        """
        return sorted(self.page_rank.items(), key=lambda x: x[1], reverse=True)

    def calcuate_CC(self):
        """
        Calculates the (undirected) clustering coefficient of each of the nodes in the graph
        :param: None
        :return: None
       ּ
         """

        for i in self.source_to_dest_dict.keys():
            e1 = len(self.source_to_dest_dict[str(i)])
            r1 = set(self.source_to_dest_dict[str(i)])

            try:
                e = e1 + len(self.dest_to_source_dict[str(i)])
                r = len(r1.union(set(self.dest_to_source_dict[str(i)])))
            except:
                e = e1
                r = len(r1)
            finally:
                if (r == 1 or r == 0):
                    self.cc[str(i)] = 0
                else:
                    self.cc[str(i)] = float(e / (abs(r) * (abs(r) - 1)))

    def get_CC(self, node_name):
        """
        Returns the clustering coefficient of a specific node.
        Return “-1” for non-existing name
        :param node_name: The node name
        :type node_name: String
        :return: The CC of the given node name
        :rtype: Double
        """
        return self.cc[node_name]

    def get_top_CC(self, n):
        """
        Returns a list of n nodes with the highest clustering coefficients.
        The list should be ordered according to the CC values from high to low.
        :param n: How many nodes
        :return: List of pairs:<node name, CC value >
        """
        return heapq.nlargest(n, self.cc.items(), key=lambda i: i[1])

    def get_all_CC(self):
        """
        Returns a list of the clustering coefficients for all the nodes in the graph
        The list should include pairs of node id, CC value of that node.
        The list should be ordered according to the CC values from high to low.
        :param: None
        :return: List of pairs:<node name, CC value >
        """
        return sorted(self.cc.items(), key=lambda x: x[1], reverse=True)

    def get_average_CC(self):
        """
        Returns a simple average over all the clustering coefficients values
        :param: None
        :return: average CC
        """
        return mean(self.cc[i] for i in self.cc)


G = graph()
G.load_graph("Wikipedia_votes.csv")
G.calculate_page_rank()
print(G.get_top_PageRank(10))