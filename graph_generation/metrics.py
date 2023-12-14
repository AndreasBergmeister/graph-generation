import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import networkx as nx
import numpy as np

# suppress networkx future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


from util.eval_helper import (
    clustering_stats,
    compute_list_eigh,
    degree_stats,
    eval_acc_planar_graph,
    eval_acc_sbm_graph,
    eval_acc_tree_graph,
    eval_fraction_isomorphic,
    eval_fraction_unique,
    eval_fraction_unique_non_isomorphic_valid,
    is_planar_graph,
    is_sbm_graph,
    orbit_stats_all,
    spectral_filter_stats,
    spectral_stats,
)


@dataclass
class Metric(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __call__(
        self,
        reference_graphs: list[nx.Graph],
        predicted_graphs: list[nx.Graph],
        train_graphs: list[nx.Graph],
    ) -> float:
        pass


class NodeNumDiff(Metric):
    def __str__(self):
        return "NodeNumDiff"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        ref_node_num = np.array([G.number_of_nodes() for G in reference_graphs])
        pred_node_num = np.array([G.number_of_nodes() for G in predicted_graphs])
        return np.mean(np.abs(ref_node_num - pred_node_num))


class NodeDegree(Metric):
    def __str__(self):
        return "Deg."

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return degree_stats(reference_graphs, predicted_graphs)


class ClusteringCoefficient(Metric):
    def __str__(self):
        return "Clus."

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return clustering_stats(reference_graphs, predicted_graphs)


class OrbitCount(Metric):
    def __str__(self):
        return "Orbit"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return orbit_stats_all(reference_graphs, predicted_graphs)


class Spectral(Metric):
    # Eigenvalues of normalized Laplacian
    def __str__(self):
        return "Spectral"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return spectral_stats(reference_graphs, predicted_graphs)


class Wavelet(Metric):
    def __str__(self):
        return "Wavelet"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        reference_eigvals, reference_eigvecs = compute_list_eigh(reference_graphs)
        predicted_eigvals, predicted_eigvecs = compute_list_eigh(predicted_graphs)
        return spectral_filter_stats(
            reference_eigvecs, reference_eigvals, predicted_eigvecs, predicted_eigvals
        )


class Ratio(Metric):
    def __init__(self):
        self.cache = {}

    def __str__(self):
        return "Ratio"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        metrics = [
            NodeDegree(),
            ClusteringCoefficient(),
            OrbitCount(),
            Spectral(),
            Wavelet(),
        ]

        # Use cached train metrics if available otherwise compute them
        reference_graphs_key = hash(tuple(reference_graphs))
        if reference_graphs_key in self.cache:
            train_metrics = self.cache[reference_graphs_key]
        else:
            train_metrics = np.array(
                [m(reference_graphs, train_graphs, train_graphs) for m in metrics]
            )
            self.cache[reference_graphs_key] = train_metrics

        predicted_metrics = np.array(
            [m(reference_graphs, predicted_graphs, train_graphs) for m in metrics]
        )

        return np.mean(predicted_metrics / (train_metrics + 1e-8))


class Uniqueness(Metric):
    def __str__(self):
        return "Uniqueness"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_fraction_unique(predicted_graphs, precise=True)


class Novelty(Metric):
    def __str__(self):
        return "Novelty"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return 1 - eval_fraction_isomorphic(
            fake_graphs=predicted_graphs, train_graphs=train_graphs
        )


class ValidPlanar(Metric):
    def __str__(self):
        return "ValidPlanar"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_acc_planar_graph(predicted_graphs)


class ValidTree(Metric):
    def __str__(self):
        return "ValidTree"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_acc_tree_graph(predicted_graphs)


class ValidSBM(Metric):
    def __str__(self):
        return "ValidSBM"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_acc_sbm_graph(predicted_graphs)


class UniqueNovelValidPlanar(Metric):
    def __str__(self):
        return "UniqueNovelValidPlanar"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_fraction_unique_non_isomorphic_valid(
            predicted_graphs, train_graphs, is_planar_graph
        )[2]


class UniqueNovelValidTree(Metric):
    def __str__(self):
        return "UniqueNovelValidTree"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_fraction_unique_non_isomorphic_valid(
            predicted_graphs, train_graphs, nx.is_tree
        )[2]


class UniqueNovelValidSBM(Metric):
    def __str__(self):
        return "UniqueNovelValidSBM"

    def __call__(self, reference_graphs, predicted_graphs, train_graphs):
        return eval_fraction_unique_non_isomorphic_valid(
            predicted_graphs, train_graphs, is_sbm_graph
        )[2]
