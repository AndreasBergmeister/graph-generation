import networkx as nx
import numpy as np
import scipy as sp


def generate_planar_graphs(num_graphs, min_size, max_size, seed=0):
    """Generate planar graphs using Delauney triangulation."""
    rng = np.random.default_rng(seed)
    graphs = []

    for _ in range(num_graphs):
        n = rng.integers(min_size, max_size, endpoint=True)
        points = rng.random((n, 2))
        tri = sp.spatial.Delaunay(points)
        adj = sp.sparse.lil_array((n, n), dtype=np.int32)
        for t in tri.simplices:
            adj[t[0], t[1]] = 1
            adj[t[1], t[2]] = 1
            adj[t[2], t[0]] = 1
            adj[t[1], t[0]] = 1
            adj[t[2], t[1]] = 1
            adj[t[0], t[2]] = 1
        G = nx.from_scipy_sparse_array(adj)
        graphs.append(G)

    return graphs


def generate_tree_graphs(num_graphs, min_size, max_size, seed=0):
    """Generate tree graphs using the networkx library."""
    rng = np.random.default_rng(seed)
    graphs = []

    for _ in range(num_graphs):
        n = rng.integers(min_size, max_size, endpoint=True)
        G = nx.random_tree(n, seed=rng)
        graphs.append(G)

    return graphs


def generate_sbm_graphs(
    num_graphs,
    min_num_communities,
    max_num_communities,
    min_community_size,
    max_community_size,
    seed=0,
):
    """Generate SBM graphs using the networkx library."""
    rng = np.random.default_rng(seed)
    graphs = []

    while len(graphs) < num_graphs:
        num_communities = rng.integers(
            min_num_communities, max_num_communities, endpoint=True
        )
        community_sizes = rng.integers(
            min_community_size, max_community_size, size=num_communities
        )
        probs = np.ones([num_communities, num_communities]) * 0.005
        probs[np.arange(num_communities), np.arange(num_communities)] = 0.3
        G = nx.stochastic_block_model(community_sizes, probs, seed=rng)
        if nx.is_connected(G):
            graphs.append(G)

    return graphs
