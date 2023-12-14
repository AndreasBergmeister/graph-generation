from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import scipy as sp
from numpy.typing import NDArray
from scipy.sparse import coo_array, csr_array, eye

real = np.floating | float


class Reduction(ABC):
    """Abstract class for graph reduction."""

    preserved_eig_size: int
    local_variation_cost: bool
    sqrt_partition_size: bool
    weighted_reduction: bool
    min_red_frac: real
    max_red_frac: real
    red_threshold: int
    rand_lambda: int

    def __init__(self, adj, lap=None, B=None, expansion_matrix=None, level=0):
        self.adj = csr_array(adj, dtype=np.float64)
        self.n = adj.shape[0]
        self.node_degree = adj.sum(0)
        self.lap = sp.sparse.diags(self.node_degree) - adj if lap is None else lap
        if B is None:
            self.B = self.get_B0()
            self.A = self.B
        else:
            self.B = B
            self.A = self.get_A()

        self.expansion_matrix = expansion_matrix
        self.node_expansion = (
            np.ones(self.n, dtype=np.int32)
            if expansion_matrix is None
            else expansion_matrix.sum(0).astype(np.int32)
        )
        self.level = level

    def get_reduced_graph(self, rng=np.random.default_rng()):
        C = self.get_coarsening_matrix(rng)
        # P_inv = C.T with all non-zero entries set to 1
        P_inv = C.T.astype(bool).astype(C.dtype)

        if self.weighted_reduction:
            lap_reduced = P_inv.T @ self.lap @ P_inv
            adj_reduced = -lap_reduced + sp.sparse.diags(lap_reduced.diagonal())
        else:
            lap_reduced = None
            adj_reduced = (P_inv.T @ self.adj @ P_inv).tocoo()
            # remove self-loops and edge weights
            row, col = adj_reduced.row, adj_reduced.col
            mask = row != col
            row, col = row[mask], col[mask]
            adj_reduced = sp.sparse.coo_array(
                (np.ones(len(row), dtype=adj_reduced.dtype), (row, col)),
                shape=adj_reduced.shape,
            )

        return self.__class__(
            adj=adj_reduced,
            lap=lap_reduced,
            B=C @ self.B,
            expansion_matrix=P_inv,
            level=self.level + 1,
        )

    def get_B0(self) -> NDArray:
        offset = 2 * np.max(self.node_degree)
        T = offset * sp.sparse.eye(self.n, format="csc") - self.lap
        lk, Uk = sp.sparse.linalg.eigsh(
            T, k=self.preserved_eig_size, which="LM", tol=1e-5
        )
        lk = (offset - lk)[::-1]
        Uk = Uk[:, ::-1]

        # compute L^-1/2
        mask = lk < 1e-5
        lk[mask] = 1
        lk_inv = 1 / np.sqrt(lk)
        lk_inv[mask] = 0
        return Uk * lk_inv[np.newaxis, :]  # = Uk @ np.diag(lk_inv)

    def get_A(self) -> NDArray:
        # A = B @ (B.T @ L @ B)^-1/2
        d, V = np.linalg.eig(self.B.T @ self.lap @ self.B)
        mask = d < 1e-8
        d[mask] = 1
        d_inv_sqrt = 1 / np.sqrt(d)
        d_inv_sqrt[mask] = 0
        return self.B @ np.diag(d_inv_sqrt) @ V

    def get_coarsening_matrix(self, rng) -> coo_array:
        # get the contraction sets and their costs
        contraction_sets = self.get_contraction_sets()
        costs = (
            np.apply_along_axis(self.get_cost, 1, contraction_sets)
            if len(contraction_sets) > 0
            else np.array([])
        )

        # compute reduction fraction
        if self.n <= self.red_threshold:
            reduction_fraction = self.max_red_frac
        else:
            reduction_fraction = rng.uniform(self.min_red_frac, self.max_red_frac)

        # get partitioning minimizing the cost in a randomized fashion
        perm = costs.argsort()
        contraction_sets = contraction_sets[perm]
        partitions = []
        marked = np.zeros(self.n, dtype=bool)
        for contraction_set in contraction_sets:
            if (
                not marked[contraction_set].any()
                and rng.uniform() >= self.rand_lambda  # randomize
            ):
                partitions.append(contraction_set)
                marked[contraction_set] = True
                if marked.sum() - len(partitions) >= reduction_fraction * self.n:
                    break

        # construct projection matrix
        P = eye(self.n, format="lil")
        mask = np.ones(self.n, dtype=bool)
        for partition in partitions:
            size = len(partition)
            size = np.sqrt(size) if self.sqrt_partition_size else size
            P[partition[0], partition] = 1 / size
            mask[partition[1:]] = False
        P = P[mask, :]
        return coo_array(P, dtype=np.float64)

    @abstractmethod
    def get_contraction_sets(self) -> Sequence[NDArray]:
        pass

    def get_cost(self, nodes: NDArray) -> real:
        if self.local_variation_cost:
            return self.get_local_variation_cost(nodes)
        else:
            return np.random.rand()

    def get_local_variation_cost(self, nodes: NDArray) -> real:
        """Compute the local variation cost for a set of nodes"""
        nc = len(nodes)
        if nc == 1:
            return np.inf

        ones = np.ones(nc)
        W = self.adj[nodes, :][:, nodes]
        L = np.diag(2 * self.node_degree[nodes] - W @ ones) - W
        B = (np.eye(nc) - np.outer(ones, ones) / nc) @ self.A[nodes, :]
        return np.linalg.norm(B.T @ L @ B) / (nc - 1)


class NeighborhoodReduction(Reduction):
    """Graph reduction by contracting neighborhoods."""

    def get_contraction_sets(self) -> Sequence[NDArray]:
        """Returns neighborhood contraction sets"""
        adj_with_self_loops = self.adj.copy().tolil()
        adj_with_self_loops.setdiag(1)
        return [np.array(nbrs) for nbrs in adj_with_self_loops.rows]


class EdgeReduction(Reduction):
    """Graph reduction by contracting edges.

    Class implements optimized routines for local variation cost computation.
    """

    def get_contraction_sets(self) -> Sequence[NDArray]:
        us, vs, _ = sp.sparse.find(sp.sparse.triu(self.adj))
        return np.stack([us, vs], axis=1)

    def get_local_variation_cost(self, edge: NDArray) -> real:
        """Compute the local variation cost for an edge"""
        u, v = edge
        w = self.adj[u, v]
        L = np.array(
            [[2 * self.node_degree[u] - w, -w], [-w, 2 * self.node_degree[v] - w]]
        )
        B = self.A[edge, :]
        return np.linalg.norm(B.T @ L @ B)


class ReductionFactory:
    def __init__(
        self,
        contraction_family,
        cost_type,
        preserved_eig_size,
        sqrt_partition_size,
        weighted_reduction,
        min_red_frac,
        max_red_frac,
        red_threshold,
        rand_lambda,
    ):
        self.contraction_family = contraction_family
        self.cost_type = cost_type
        self.preserved_eig_size = preserved_eig_size
        self.sqrt_partition_size = sqrt_partition_size
        self.weighted_reduction = weighted_reduction
        self.min_red_frac = min_red_frac
        self.max_red_frac = max_red_frac
        self.red_threshold = red_threshold
        self.rand_lambda = rand_lambda

    def __call__(self, adj: NDArray) -> Reduction:
        if self.contraction_family == "neighborhoods":
            reduction = NeighborhoodReduction
        elif self.contraction_family == "edges":
            reduction = EdgeReduction
        else:
            raise ValueError("Unknown contraction family.")

        if self.cost_type == "local_variation":
            reduction.local_variation_cost = True
        elif self.cost_type == "random":
            reduction.local_variation_cost = False
        else:
            raise ValueError("Unknown reduction cost type.")

        reduction.preserved_eig_size = self.preserved_eig_size
        reduction.sqrt_partition_size = self.sqrt_partition_size
        reduction.weighted_reduction = self.weighted_reduction
        reduction.min_red_frac = self.min_red_frac
        reduction.max_red_frac = self.max_red_frac
        reduction.red_threshold = self.red_threshold
        reduction.rand_lambda = self.rand_lambda

        return reduction(adj)
