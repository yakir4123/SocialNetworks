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

th = 0
NODE_SIZE = 70
REGENERATE_DATA = False
OUTPUT_DIR = f'outputs{os.sep}'

roles = {}


def add_edges(G: nx.Graph, cliqe_nodes: typing.Iterable, weight: int):
    for u, v in itertools.combinations(cliqe_nodes, 2):
        if G.has_edge(u, v):
            G[u][v]['weight'] += weight
            G[u][v]['matches'] += 1
        else:
            G.add_edge(u, v, weight=weight)
            G[u][v]['matches'] = 1
            G.nodes[u]['lane'] = roles[u]['lane']
            G.nodes[u]['role'] = roles[u]['role']


def remove_edges(G: nx.Graph, th):
    to_remove = [(u, v) for u, v, C in G.edges.data('matches') if C < th]
    [G.remove_edge(u, v) for u, v in to_remove]


def plot_th(G: nx.Graph, thresholds: typing.Iterable):
    Y = []
    X = list(thresholds)
    for th in X:
        remove_edges(G, th)
        Y.append(G.number_of_edges())
    nodes = G.number_of_nodes()
    plt.plot(X, Y, 'g', label='number of edges')
    plt.plot(X, [nodes * (nodes - 1) / 2 - y for y in Y], 'r', label='missing edges')
    plt.legend()
    plt.show()


def generate_champions_graph(matches_files_names: typing.Iterable[str], th: int):
    champions_graph = nx.Graph()

    # to avoid cases of count match multiple times
    visited_matches = []
    print('iterating over all the files')
    with alive_bar(len(matches_files_names), force_tty=True) as bar:
        for file_name in matches_files_names:
            bar()
            with open(f'matches{os.sep}{file_name}', "r") as f:
                try:
                    for match in json.load(f):
                        if match['gameId'] in visited_matches:
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
                except (UnicodeDecodeError, json.decoder.JSONDecodeError):
                    continue

    # remove self loops
    champions_graph.remove_edges_from(nx.selfloop_edges(champions_graph))

    # remove outliers
    remove_edges(champions_graph, th)
    return champions_graph, visited_matches


def plot_graph(G: nx.Graph):
    fig, ax = plt.subplots()
    ax.title.set_text('Weighted Champions Graph')
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, with_labels=True, font_weight='bold', node_size=NODE_SIZE)
    nx.write_adjlist(G, f"{OUTPUT_DIR}champions_graph.adjlist")
    plt.show()


def role_to_id(matches_files_names: typing.Iterable[str]) -> typing.Dict:
    with alive_bar(len(matches_files_names), force_tty=True) as bar:
        for file_name in matches_files_names:
            bar()
            with open(f'matches{os.sep}{file_name}', "r") as f:
                try:
                    for match in json.load(f):
                        for participant in match["participants"]:
                            if roles.get(participant['championId']) is not None:
                                continue
                            roles[participant['championId']] = \
                                {'role': participant['timeline']['role'], 'lane': participant['timeline']['lane']}
                except (UnicodeDecodeError, json.decoder.JSONDecodeError):
                    continue
    return roles


def create_train_df(G: nx.Graph, features: typing.Dict[str, typing.Callable]) -> pd.DataFrame:
    df = pd.DataFrame(columns=features.keys())

    print('iterating over all the edges')
    with alive_bar(len(G.edges), force_tty=True) as bar:
        backup_progress = 0
        for u, v in G.edges:

            row = {}
            for feature_name, feature_func in features.items():
                row[feature_name] = feature_func(u, v)
            df.loc[len(df)] = row
            bar()
            backup_progress += 1
            if backup_progress == len(G.edges) // 10:
                backup_progress = 0
                df.to_csv(f'{OUTPUT_DIR}data_features_bup_{u}_{v}.csv', index=False)
    return df


def regenerate_graph(th: int, verbose=False):
    files_name = os.listdir(os.path.dirname(os.path.abspath(__file__)) + os.sep + "matches")

    G, matches = generate_champions_graph(files_name, th)
    n_nodes = G.number_of_nodes()

    if verbose:
        print(f'Num of matches: {len(matches)}')
        print(f'Num of nodes: {n_nodes}')
        # u,v and v,u is counted twice
        print(f'Num of edges: {G.number_of_edges()}')
        print(f'Edges with zero weight: {sum(1 if w == 0 else 0 for u, v, w in G.edges.data("weight"))}')
        print(f'Num of maximum edges in graph: {n_nodes * (n_nodes - 1) / 2}')
        print(f'Missing edges: {(n_nodes * (n_nodes - 1) / 2) - G.number_of_edges()}')
        plot_graph(G)

    return G


if __name__ == '__main__':
    # Create features
    feature_extractor = ff.FeatureExtractor()

    features = {'W(u)': lambda u, v: feature_extractor.node_weight(u),
                'W(v)': lambda u, v: feature_extractor.node_weight(v),
                # 'd5': lambda u, v: feature_extractor.avarage_k_path_weight(u, v, 5), # just to make the feature extracator the paths that already calculated
                # 'd4': lambda u, v: feature_extractor.avarage_k_path_weight(u,   v, 4),
                'd3': lambda u, v: feature_extractor.avarage_k_path_weight(u, v, 3),
                # 'var5': lambda u, v: feature_extractor.var_k_path_weight(u, v, 5),
                # 'var4': lambda u, v: feature_extractor.var_k_path_weight(u, v, 4),
                'var3': lambda u, v: feature_extractor.var_k_path_weight(u, v, 3),
                'w_centrality': lambda u, v: feature_extractor.edge_weighted_betweenness_centrality(u, v),
                'uw_centrality': lambda u, v: feature_extractor.edge_unweighted_betweenness_centrality(u, v),
                # 'role(u)': lambda u, v: feature_extractor.role(u),
                # 'role(v)': lambda u, v: feature_extractor.role(v),
                # 'lane(u)': lambda u, v: feature_extractor.lane(u),
                # 'lane(v)': lambda u, v: feature_extractor.lane(v),
                'sign': lambda u, v: feature_extractor.sign(u, v),
                'score': lambda u, v: feature_extractor.score(u, v)}
    for th in [0]:
        if REGENERATE_DATA:
            files_name = os.listdir(os.path.dirname(os.path.abspath(__file__)) + os.sep + "matches")
            roles_df = role_to_id(files_name)
            G = regenerate_graph(th, True)
            feature_extractor.setG(G)
            df = create_train_df(G, features=features)
            df.to_csv(f'{OUTPUT_DIR}data_features_{th}_roles.csv', index=False)
        else:
            df = pd.read_csv(f'{OUTPUT_DIR}data_features_{th}_roles.csv')
        # df[list(features.keys())] = df[list(features.keys())].astype('int16')
        df = pd.concat([df, pd.get_dummies(df['role(u)'], prefix='role(u)')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['role(v)'], prefix='role(v)')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['lane(u)'], prefix='lane(u)')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['lane(v)'], prefix='lane(v)')], axis=1)
        df.drop(['role(u)', 'role(v)', 'lane(u)', 'lane(v)'], axis=1, inplace=True)

        cols_to_use = ['d3', 'var3', 'w_centrality', 'uw_centrality', 'role(u)', 'role(v)', 'lane(u)', 'lane(v)']
        # cols_to_use = list(df.columns)
        # cols_to_use.remove('sign')
        # cols_to_use.remove('score')
        # print(df.describe())
        # Select subset of predictors
        X = df[cols_to_use]

        # Select target
        y = df.sign

        # Separate data into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

        model = XGBRegressor()
        model.fit(X_train, y_train)

        sign_valid = np.sign(y_valid)
        predictions = model.predict(X_valid)
        sign_predictions = np.sign(predictions)

        print(f"threshold = {th}")
        print("Mean Sign Absolute Error: " + str(mean_absolute_error(sign_predictions, sign_valid)))
        print(
            f"{np.sum(sign_valid.to_numpy(dtype='int16') == sign_predictions) / len(sign_predictions) * 100} of success")
    print('Done.')
