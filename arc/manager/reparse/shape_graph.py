import networkx as nx
from ...base import *
from ...graphic import *
from dataclasses import dataclass

Edge = Union[tuple[int, int, dict], tuple[int, int]]


class ShapeGraph:
    def __init__(self, all_all_nodes: list[list[list[Shape]]],
                 edges: list[Edge], di_graph: bool = False)->None:
        self.graph = nx.DiGraph() if di_graph else nx.Graph()
        self.di_graph = di_graph
        id_checker = set()

        for all_nodes in all_all_nodes:
            for nodes in all_nodes:
                for node in nodes:
                    _id = id(node)
                    self.graph.add_node(_id, obj=node)
                    id_checker.add(_id)

        for edge in edges:
            assert edge[0] in id_checker
            assert edge[1] in id_checker
        self.graph.add_edges_from(edges)  # type:ignore

    def lookup(self, shape: Shape)->list[tuple[Shape, dict]]:
        result = []
        for nbr, data in self.graph.adj[id(shape)].items():
            result.append((self.graph.nodes[nbr]['obj'], data))
        return result

    def number_of_edges(self)->int:
        return self.graph.number_of_edges()

    def number_of_nodes(self)->int:
        return self.graph.number_of_nodes()

    def connected_components(self)->tuple[dict[int, int], int]:
        '''
        Find and return connected components.
        The result format is {shape_id: component_id}, component_count.
        '''

        result = {}
        cluster_id = 0
        for cluster in nx.connected_components(self.graph):
            cluster_id += 1

            for node in cluster:
                result[node] = cluster_id
        return result, cluster_id

    def __repr__(self)->str:
        n_nodes, n_edges = self.number_of_nodes(), self.number_of_edges()
        return f'ShapeGraph(n_nodes={n_nodes},n_edges={n_edges})\n'
