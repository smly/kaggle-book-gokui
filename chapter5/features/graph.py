import networkx as nx
import numpy as np
import pandas as pd

from common.constants import FEATURE_MEMORY
from texts.preprocessing import PreprocessingKey, get_dataset


def build_graph(df):
    g = nx.Graph()
    for q1, q2, is_duplicate in zip(
        df["question1"], df["question2"], df["is_duplicate"]
    ):
        g.add_edge(q1, q2, is_test=np.isnan(is_duplicate))
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def graph_link_prediction_features(df):
    link_prediction_scorer = [
        nx.resource_allocation_index,
        nx.jaccard_coefficient,
        nx.preferential_attachment,
        nx.adamic_adar_index,
    ]

    g = build_graph(df)
    edges = [(q1, q2) for q1, q2 in zip(df.question1, df.question2)]
    edges_wo_self_loops = [(q1, q2) for q1, q2 in edges if q1 != q2]
    features = pd.DataFrame()
    for link_prediction_scorer in link_prediction_scorer:
        score_map = {
            (q1, q2): score
            for q1, q2, score in link_prediction_scorer(g, edges_wo_self_loops)
        }
        features[link_prediction_scorer.__name__] = [
            score_map.get((q1, q2), 0) for q1, q2 in edges
        ]
    return features


@FEATURE_MEMORY.cache
def build_graph_link_prediction_features():
    df = get_dataset(PreprocessingKey.SPACE_TOKENIZATION)
    return graph_link_prediction_features(df)


@FEATURE_MEMORY.cache
def build_graph_node_features():
    node_scorer = [
        nx.core_number,
        nx.clustering,
        nx.square_clustering,
        nx.pagerank,
    ]

    df = get_dataset(PreprocessingKey.SPACE_TOKENIZATION)
    g = build_graph(df)

    features = pd.DataFrame()
    for node_scorer in node_scorer:
        node_scores = node_scorer(g)
        features[f"{node_scorer.__name__}_q1"] = df["question1"].map(
            node_scores
        )
        features[f"{node_scorer.__name__}_q2"] = df["question2"].map(
            node_scores
        )
    return features


@FEATURE_MEMORY.cache
def build_graph_connected_component_features():
    df = get_dataset(PreprocessingKey.SPACE_TOKENIZATION)
    g = build_graph(df)

    cc_node_sizes = {}
    cc_edge_sizes = {}
    cc_densities = {}
    cc_edge_test_ratios = {}
    for i, cc in enumerate(nx.connected_components(g)):
        sub_g: nx.Graph = g.subgraph(cc)
        cc_node_size = len(cc)
        cc_edge_size = sub_g.number_of_edges()
        cc_density = cc_edge_size / cc_node_size
        cc_edge_test_ratio = 0
        for _, _, data in sub_g.edges(data=True):
            cc_edge_test_ratio = int(data["is_test"]) / cc_edge_size

        for node in cc:
            cc_node_sizes[node] = cc_node_size
            cc_edge_sizes[node] = cc_edge_size
            cc_densities[node] = cc_density
            cc_edge_test_ratios[node] = cc_edge_test_ratio

    features = pd.DataFrame()
    features["cc_node_size"] = df["question1"].map(cc_node_sizes)
    features["cc_edge_size"] = df["question1"].map(cc_edge_sizes)
    features["cc_density"] = df["question1"].map(cc_densities)
    features["cc_edge_test_ratio"] = df["question1"].map(cc_edge_test_ratios)
    return features
