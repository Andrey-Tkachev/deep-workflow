import random

import networkx as nx
import numpy as np
import torch


def force_weakly_connected(graph: nx.DiGraph):
    components = list(map(min, nx.weakly_connected_components(graph)))
    root = components[0]
    for sub_component in components[1:]:
        G.add_edge(root, sub_component)


def generate_random_dag(n, p, weight_range=(0, 100)):
    random_graph = nx.fast_gnp_random_graph(n, p, directed=True)
    random_weights = np.array(
        [random.randint(*weight_range) for _ in range(n)], dtype=float)
    random_dag = nx.DiGraph([
        (u, v) for (u, v) in random_graph.edges() if u < v
    ])

    zero_degree = [n for n in random_dag.nodes() if random_dag.degree[n] == 0]
    mask = np.ones((n, ))
    mask[zero_degree] = 0.0
    random_dag.remove_nodes_from(zero_degree)

    if not nx.is_weakly_connected(random_dag):
        force_weakly_connected(random_dag)
    
    nx.set_node_attributes(
        random_dag,
        {
            node: {
                'weight': float(weight),
                'features': np.array([
                    float(weight),
                    float(len(nx.descendants(random_dag, node))),
                    float(len(nx.ancestors(random_dag, node))),
                ]).reshape((1, -1)) if mask[node] > 0.0 else None,
            } for node, weight in enumerate(random_weights)
        }
    )

    return random_dag, mask


def calculate_critical_value(graph: nx.DiGraph, mask: np.array):
    '''Calcualte critical value for each node in graph

    Critical value of node u is max({critical_value[v]: v is child of u}) + weight[u]
    '''
    critical_values = np.zeros_like(mask, dtype=float)
    for node in reversed(list(nx.topological_sort(graph))):
        critical_values[node] = graph.nodes[node]['weight']
        descendants = nx.descendants(graph, node)
        if descendants:
            critical_values[node] += max(map(
                lambda v: critical_values[v],
                descendants
            ))
    return critical_values


def create_dag_dataset(size=100, nodes_num=100, edge_prob=0.2, weight_range=(0, 100)):
    dags = []
    masks = []
    critical_values = []
    for _ in range(size):
        dag, nodes_mask = generate_random_dag(nodes_num, edge_prob, weight_range=weight_range)
        dags.append(dag)
        masks.append(nodes_mask)
        critical_values.append(
            calculate_critical_value(dag, nodes_mask)
        )
    return dags, torch.tensor(masks, dtype=torch.float), torch.tensor(critical_values, dtype=torch.float)
