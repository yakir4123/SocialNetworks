import os
import json
import typing
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from alive_progress import alive_bar
from LolChampions import features_funcs as ff
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

NODE_SIZE = 70
REGENERATE_DATA = False
OUTPUT_DIR = f'outputs{os.sep}'


def add_edges(G: nx.Graph, cliqe_nodes: typing.Iterable, weight: int):
    for u, v in itertools.combinations(cliqe_nodes, 2):
        if G.has_edge(u, v):
            G[u][v]['weight'] += weight
        else:
            G.add_edge(u, v, weight=weight)


def generate_champions_graph(matches_files_names: typing.Iterable[str]):
    champions_graph = nx.Graph()
    matches_data = []
    for file_name in matches_files_names:
        with open(f'matches{os.sep}{file_name}', "r") as f:
            matches_data = matches_data + json.load(f)

    # to avoid cases of count match multiple times
    visited_matches = []
    for match in matches_data:
        if match in visited_matches:
            continue
        visited_matches.append(match['gameId'])
        win_cid = []
        loss_cid = []
        for participant in match["participants"]:
            try:
                if participant['stats']['win']:
                    win_cid.append(participant['championId'])
                else:
                    loss_cid.append(participant['championId'])
            except KeyError:
                # In cases that the data is missing
                pass

        add_edges(champions_graph, win_cid, 1)
        add_edges(champions_graph, loss_cid, -1)
    return champions_graph, visited_matches


def plot_graph(G: nx.Graph):
    fig, ax = plt.subplots()
    ax.title.set_text('Weighted Champions Graph')
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, with_labels=True, font_weight='bold', node_size=NODE_SIZE)
    nx.write_adjlist(G, f"{OUTPUT_DIR}champions_graph.adjlist")
    plt.show()


def set_edges_sign(G: nx.Graph):
    for u, v, w in G.edges.data('weight'):
        if w > 0:
            G[u][v]['color'] = 'green'
        elif w < 0:
            G[u][v]['color'] = 'red'
        else:  # == 0
            G[u][v]['color'] = 'black'


def create_train_df(G: nx.Graph, features: typing.Dict[str, typing.Callable]) -> pd.DataFrame:
    df = pd.DataFrame(columns=features.keys())
    with alive_bar(len(G.edges), force_tty=True) as bar:
        for u, v in G.edges:
            row = {}
            for feature_name, feature_func in features.items():
                row[feature_name] = feature_func(G, u, v)
            df.loc[len(df)] = row
            bar()
    return df


def regenerate_data():
    files_name = ['ohjaja_100.json', 'ohjjaju_100.json',
                  'sewip_100.json', 'The sammoner_100.json',
                  'troiiing hard_100.json', 'kamikaz123_100.json',
                  'RidicuIous_100.json']

    G, matches = generate_champions_graph(files_name)
    set_edges_sign(G)
    n_nodes = G.number_of_nodes()
    print(f'Num of matches: {len(matches)}')
    print(f'Num of nodes: {G.number_of_nodes()}')
    print(f'Num of edges: {G.number_of_edges()}')
    print(f'Edges with zero weight: {sum(1 if w == 0 else 0 for u, v, w in G.edges.data("weight"))}')
    print(f'Num of maximum edges in graph: {n_nodes * (n_nodes - 1) / 2}')
    print(f'Missing edges: {n_nodes * (n_nodes - 1) / 2 - G.number_of_edges()}')
    print(f"'+' sign edges: {sum(1 if c == 'green' else 0 for u, v, c in G.edges.data('color'))}")
    print(f"'-' sign edges: {sum(1 if c == 'red' else 0 for u, v, c in G.edges.data('color'))}")
    print(f"'0' weighted edges: {sum(1 if c == 'white' else 0 for u, v, c in G.edges.data('color'))}")
    plot_graph(G)

    features = {'W(u)': lambda G, u, v: ff.node_weight(G, u),
                'W(v)': lambda G, u, v: ff.node_weight(G, v),
                'd2': lambda G, u, v: ff.avarage_k_path_weight(G, u, v, 2),
                'd3': lambda G, u, v: ff.avarage_k_path_weight(G, u, v, 3),
                'd4': lambda G, u, v: ff.avarage_k_path_weight(G, u, v, 4),
                # 'd5': lambda G, u, v:ff.avarage_k_path_weight(G, u, v, 5),
                'std2': lambda G, u, v: ff.std_k_path_weight(G, u, v, 2),
                'std3': lambda G, u, v: ff.std_k_path_weight(G, u, v, 3),
                'std4': lambda G, u, v: ff.std_k_path_weight(G, u, v, 4),
                # 'std5': lambda G, u, v:ff.std_k_path_weight(G, u, v, 5),
                'weight': lambda G, u, v: G[u][v]['weight'],
                'sign': lambda G, u, v: ff.sign(G, u, v)}
    df = create_train_df(G, features=features)
    df.to_csv(f'{OUTPUT_DIR}data_features.csv', index=False)
    return df


if __name__ == '__main__':

    if REGENERATE_DATA:
        df = regenerate_data()
    else:
        df = pd.read_csv(f'{OUTPUT_DIR}data_features.csv')

    print(df.describe())
    # Select subset of predictors
    cols_to_use = ['W(u)', 'W(v)', 'd2', 'd3', 'd4', 'std2', 'std3', 'std4']
    X = df[cols_to_use]

    # Select target
    y = df.sign

    # Separate data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = XGBRegressor()
    model.fit(X_train, y_train)

    sign_valid = np.sign(y_valid)
    predictions = model.predict(X_valid)
    sign_predictions = np.sign(predictions)

    print("Mean Sign Absolute Error: " + str(mean_absolute_error(sign_predictions, sign_valid)))
    print('Done.')
