from .abstract_modeling import *
import networkx as nx
from typing import Generator
import itertools


class PlanningGraph:
    '''
    Directed graph with node as state and edge as transition.
    Each edge contains a list of Tasks and Actions required for state transition.
    '''

    def __init__(self, start_state: State)->None:
        self.graph = nx.DiGraph()
        self.graph.add_node(start_state)
        self.start_state = start_state
        self.end_states = []

    def add_state(self, before: State, after: State, task: Task, action: Action,
                  is_end: bool)->bool:
        '''Add new state to the graph and return True if new edge is created.'''
        assert before in self.graph
        if before == after:
            return False

        edge_data = self.graph.get_edge_data(before, after, default=None)
        if edge_data is None:
            self.graph.add_edge(before, after, data=[(task, action)])
            if is_end:
                self.end_states.append(after)
            return True
        else:
            edge_data['data'].append((task, action))
            return False

    def shortest_simple_paths(self)->Generator[list[State], None, None]:
        for end_state in self.end_states:
            for path in nx.shortest_simple_paths(
                    self.graph, self.start_state, end_state):
                yield path

    def get_edge_data(self, before: State, after: State)->list[tuple[Task, Action]]:
        result = self.graph.get_edge_data(before, after, default={'data': []})
        return result['data']

    def trim(self)->None:
        selected_nodes = set()
        for end_state in self.end_states:
            for path in nx.all_simple_paths(
                    self.graph, source=self.start_state, target=end_state):
                selected_nodes.update(path)
        removing_nodes = [node for node in self.graph if node not in selected_nodes]
        self.graph.remove_nodes_from(removing_nodes)

    def __repr__(self)->str:
        result = ['PlanningGraph{',
                  f'node_count: {self.graph.number_of_nodes()}',
                  f'edge_count: {self.graph.number_of_edges()}',
                  f'terminal_count: {len(self.end_states)}']
        count = 0
        for end_state in self.end_states:
            for path in nx.all_simple_paths(
                    self.graph, source=self.start_state, target=end_state):
                count += 1
                result.append(f'\n  path #{count}')
                for i, node in enumerate(path[:-1]):
                    result.append(f'  {self.get_edge_data(node, path[i+1])}')
        return '\n'.join(result+['}'])
