import heapq 
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional, DefaultDict, Iterator
from dataclasses import dataclass
from math import isclose, fsum, log
from functools import total_ordering

@total_ordering
@dataclass(frozen=True)
class Edge:
    """Immutable edge representation with canonical ordering and comparison."""
    u: str
    v: str
    weight: float

    def __new__(cls, u: str, v: str, weight: float) -> 'Edge':
        self = object.__new__(cls)  # Correctly use object.__new__
        # Use object.__setattr__ to set attributes during initialization
        # This is safe because the instance isn't fully constructed yet
        object.__setattr__(self, 'u', u if u <= v else v)
        object.__setattr__(self, 'v', v if u <= v else u)
        object.__setattr__(self, 'weight', weight)
        return self

    def __iter__(self) -> Iterator[str]:
        yield self.u
        yield self.v

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return (self.weight, self.u, self.v) < (other.weight, other.u, other.v)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return (self.u, self.v, self.weight) == (other.u, other.v, other.weight)

    def __hash__(self) -> int:
        return hash((self.u, self.v, self.weight))

@dataclass
class Triangle:
    """Represents a triangle with its edges and p-mean weight."""
    edges: Tuple[Edge, ...]
    weight: float
    
    def __post_init__(self):
        if len(self.edges) != 3:
            raise ValueError("Triangle must have exactly 3 edges")
        self.vertices = tuple(sorted(set(v for edge in self.edges for v in (edge.u, edge.v))))
        if len(self.vertices) != 3:
            raise ValueError("Triangle must have exactly 3 distinct vertices")

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Triangle):
            return NotImplemented
        return self.weight < other.weight

class DynamicHeavyLightAlgorithm:
    """Improved implementation of Dynamic Heavy-Light Algorithm for finding top-k triangles."""
    
    MAX_ITERATIONS = 1000  # Maximum number of iterations for convergence
    CONVERGENCE_TOLERANCE = 1e-9  # Relative tolerance for convergence checks
    MAX_WEIGHT_RATIO = 1e308  # Maximum allowed ratio between weights to prevent overflow

    def __init__(self, graph: Dict[str, List[Tuple[str, float]]], p: float, k: int, alpha_p: float):
        """
        Initialize the Dynamic Heavy-Light Algorithm with improved data structures.
        
        Args:
            graph: Dictionary representing weighted undirected graph
            p: Parameter for p-mean calculation (must be > 0)
            k: Number of top triangles to find (must be > 0)
            alpha_p: Threshold parameter for edge promotion (0 < alpha_p < 1)
            
        Raises:
            ValueError: If parameters are invalid or graph is too small
        """
        self._validate_parameters(p, k, alpha_p)
        self.edges, self.adj_list = self._normalize_graph(graph)
        self._validate_graph()
        
        self.p = p
        self.k = k
        self.alpha_p = alpha_p
        self.threshold = float('-inf')
        self.top_k_triangles: List[Triangle] = []
        
        # Cache for adjacent vertices
        self.vertex_neighbors: DefaultDict[str, Set[str]] = defaultdict(set)
        self._build_vertex_neighbors()

    def _validate_parameters(self, p: float, k: int, alpha_p: float) -> None:
        """Validate input parameters."""
        if p <= 0:
            raise ValueError("p must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        if not 0 < alpha_p < 1:
            raise ValueError("alpha_p must be between 0 and 1")

    def _validate_graph(self) -> None:
        """Validate graph structure."""
        if len(self.adj_list) < 3:
            raise ValueError("Graph must have at least 3 vertices")

    def _build_vertex_neighbors(self) -> None:
        """Build cache of adjacent vertices for faster triangle checking."""
        for edge in self.edges:
            self.vertex_neighbors[edge.u].add(edge.v)
            self.vertex_neighbors[edge.v].add(edge.u)

    def _normalize_graph(self, graph: Dict[str, List[Tuple[str, float]]]) -> Tuple[List[Edge], DefaultDict[str, Dict[str, float]]]:
        """
        Normalize graph to canonical edge representation and build adjacency list.
        
        Returns:
            Tuple of (sorted edge list, adjacency dictionary)
        """
        edges = set()
        adj_list: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        
        for u, neighbors in graph.items():
            for v, weight in neighbors:
                if weight <= 0:
                    raise ValueError(f"Edge weight must be positive: ({u}, {v}, {weight})")
                
                edge = Edge(u, v, weight)
                edges.add(edge)
                adj_list[edge.u][edge.v] = weight
                adj_list[edge.v][edge.u] = weight
        
        return sorted(edges, reverse=True), adj_list

    def p_mean_weight(self, edges: Tuple[Edge, ...]) -> float:
        """
        Calculate p-mean weight of triangle edges with improved numerical stability.
        Uses log-space calculations for high values of p to prevent overflow.
        """
        if len(edges) != 3:
            return float('-inf')
        
        weights = [e.weight for e in edges]
        max_weight = max(weights)
        min_weight = min(weights)
        
        if max_weight == 0 or min_weight == 0:
            return 0
            
        # Check if weight ratios are too extreme
        if max_weight / min_weight > self.MAX_WEIGHT_RATIO:
            return float('-inf')
            
        try:
            if self.p > 50:  # Use log-space for high p values
                log_weights = [log(w) for w in weights]
                log_sum = log(fsum(exp(p * (lw - max(log_weights))) 
                                 for lw in log_weights))
                return exp((log_sum - log(3)) / self.p + max(log_weights))
            else:
                # Use normalized weights and fsum for better precision
                normalized = [w / max_weight for w in weights]
                power_sum = fsum(w ** self.p for w in normalized)
                return max_weight * (power_sum / 3) ** (1/self.p)
        except (OverflowError, ZeroDivisionError):
            return float('-inf')

    def _forms_valid_triangle(self, e1: Edge, e2: Edge, e3: Edge) -> bool:
        """
        Efficiently check if three edges form a valid triangle using cached neighbors.
        """
        vertices = {e1.u, e1.v, e2.u, e2.v, e3.u, e3.v}
        if len(vertices) != 3:
            return False
            
        v1, v2, v3 = sorted(vertices)
        return (v2 in self.vertex_neighbors[v1] and 
                v3 in self.vertex_neighbors[v1] and 
                v3 in self.vertex_neighbors[v2])

    def _update_threshold(self) -> float:
        """Update threshold based on current minimum weight in top-k."""
        if not self.top_k_triangles:
            return float('-inf')
        min_weight = self.top_k_triangles[0].weight  # Heap stores minimum at index 0
        return min_weight * (1 - self.alpha_p) if min_weight != float('-inf') else float('-inf')

    def _add_triangle(self, edges: Tuple[Edge, ...], weight: float) -> bool:
        """
        Add triangle to top-k collection if it meets criteria.
        Returns True if triangle was added/updated.
        """
        if weight <= self.threshold:
            return False
            
        triangle = Triangle(edges, weight)
        
        if len(self.top_k_triangles) < self.k:
            heapq.heappush(self.top_k_triangles, triangle)
            if len(self.top_k_triangles) == self.k:
                self.threshold = self._update_threshold()
            return True
            
        if triangle.weight > self.top_k_triangles[0].weight:
            heapq.heapreplace(self.top_k_triangles, triangle)
            self.threshold = self._update_threshold()
            return True
            
        return False

    def _process_candidate_edges(self, edge: Edge, partition: Set[Edge]) -> bool:
        """
        Process a newly promoted edge against existing edges in partition.
        Uses cached neighbor information for efficient triangle detection.
        """
        modified = False
        # More efficient set comprehension
        candidates = {e for e in partition 
                     if any(v in self.vertex_neighbors[u]
                           for u in (edge.u, edge.v)
                           for v in (e.u, e.v))}
        
        # Convert to list once for iteration
        candidate_list = list(candidates)
        for i, e1 in enumerate(candidate_list):
            for e2 in candidate_list[i+1:]:
                if self._forms_valid_triangle(edge, e1, e2):
                    weight = self.p_mean_weight((edge, e1, e2))
                    if self._add_triangle((edge, e1, e2), weight):
                        modified = True
                    
        return modified

    def find_top_k_triangles(self) -> List[Triangle]:
        """
        Find top-k triangles using optimized dynamic heavy-light partitioning.
        Returns triangles sorted by weight in descending order.
        """
        if not self.edges:
            return []

        S: Set[Edge] = set()  # Super-heavy edges
        H: Set[Edge] = set()  # Heavy edges
        L: Set[Edge] = set(self.edges)  # Light edges
        
        iteration = 0
        prev_threshold = float('-inf')
        
        while L and iteration < self.MAX_ITERATIONS:
            modified = False
            
            # Process edges near threshold for potential promotion
            threshold_value = (self.threshold * self.alpha_p 
                             if self.threshold != float('-inf') 
                             else float('-inf'))
            threshold_candidates = {e for e in L if e.weight > threshold_value}
            
            for edge in threshold_candidates:
                L.remove(edge)
                if len(S) < self.k and edge.weight > self.threshold:
                    # Promote to S
                    S.add(edge)
                    if self._process_candidate_edges(edge, L):
                        modified = True
                else:
                    # Promote to H
                    H.add(edge)
                    if self._process_candidate_edges(edge, S.union(H)):
                        modified = True
            
            # Check convergence
            if (not modified and not threshold_candidates or
                isclose(self.threshold, prev_threshold, 
                       rel_tol=self.CONVERGENCE_TOLERANCE)):
                break
                
            prev_threshold = self.threshold
            iteration += 1

        # Use heapq.nlargest for efficient top-k extraction
        return heapq.nlargest(self.k, self.top_k_triangles, key=lambda t: t.weight)

def test_algorithm():
    """Test the algorithm with various cases."""
    test_graphs = [
        {  # Complete graph with 4 vertices
            'A': [('B', 10), ('C', 8), ('D', 5)],
            'B': [('A', 10), ('C', 7), ('D', 6)],
            'C': [('A', 8), ('B', 7), ('D', 4)],
            'D': [('A', 5), ('B', 6), ('C', 4)]
        },
        {  # Triangle with pendant
            'A': [('B', 5), ('C', 4)],
            'B': [('A', 5), ('C', 3), ('D', 1)],
            'C': [('A', 4), ('B', 3)],
            'D': [('B', 1)]
        }
    ]
    
    for i, graph in enumerate(test_graphs, 1):
        try:
            print(f"\nTest case {i}:")
            algorithm = DynamicHeavyLightAlgorithm(
                graph=graph,
                p=1,
                k=3,
                alpha_p=0.8
            )
            
            top_k_triangles = algorithm.find_top_k_triangles()
            
            print(f"Found {len(top_k_triangles)} triangles:")
            for triangle in top_k_triangles:
                print(f"Weight: {triangle.weight:.2f}")
                print(f"Vertices: {', '.join(triangle.vertices)}")
                print(f"Edge weights: {[edge.weight for edge in triangle.edges]}\n")
                
        except ValueError as e:
            print(f"Error processing graph {i}: {e}")

if __name__ == "__main__":
    test_algorithm()
