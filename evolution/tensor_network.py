import itertools

import networkx as nx
import random
from matplotlib import pyplot as plt
from evolution import tensor_node


class TensorNetwork:
    id_iter = itertools.count(100)

    def __init__(self, input_shapes: list, output_units: list):
        self.graph = nx.DiGraph()
        self.net_id = next(TensorNetwork.id_iter)
        self.all_nodes = {}
        self.input_ids = []
        self.output_ids = []

        self.create_inputs(input_shapes)
        self.create_outputs(output_units)

    def create_inputs(self, input_shapes: list):
        if len(input_shapes) > 1:
            raise ValueError("Multiple inputs not yet supported")

        for shape in input_shapes:
            node = tensor_node.InputNode(shape)
            self.register_node(node, "input")

    def create_outputs(self, output_units: list):
        for units in output_units:
            node = tensor_node.OutputNode(units)
            self.register_node(node, "output")

            for input_id in self.input_ids:
                self.graph.add_edge(input_id, node.id)

    def insert_node(self, new_node: tensor_node.TensorNode, position: int):
        # inserts node before the given position
        # positions refer to the index in the "non_input_nodes" list, which is kept in no particular oder
        nodes = self.get_not_input_nodes()
        if position > (len(nodes) - 1):
            raise ValueError("Invalid request to insert node: Length = " +
                             str(len(nodes)) + " position given as: " + str(position))

        self.register_node(new_node, "other")
        child_node = list(nodes.values())[position]  # TensorNode

        parents = self.get_parents(child_node)

        for parent in parents:
            self.graph.remove_edge(parent.id, child_node.id)
            self.graph.add_edge(parent.id, new_node.id)

        self.graph.add_edge(new_node.id, child_node.id)

    def delete_node(self, position=None, node_id=None):
        node_to_remove = self.get_a_middle_node(position, node_id)

        parents = self.get_parents(node_to_remove)
        children = self.get_children(node_to_remove)

        self.all_nodes.pop(node_to_remove.id)

        while node_to_remove.id in self.graph.nodes:
            self.graph.remove_node(node_to_remove.id)  # also removes adjacent edges

        for parent in parents:
            for child in children:
                self.graph.add_edge(parent.id, child.id)

    def register_node(self, node: tensor_node.TensorNode, node_type: str = None):
        self.all_nodes[node.id] = node
        self.graph.add_node(node.id, label=node.get_label())

        if node_type == "input":
            self.input_ids.append(node.id)
        elif node_type == "output":
            self.output_ids.append(node.id)

    def replace_node(self, replacement_node: tensor_node.TensorNode,
                     position=None, node_id=None):

        old_node = self.get_a_middle_node(position, node_id)
        self.all_nodes.pop(old_node.id)
        self.all_nodes[replacement_node.id] = replacement_node

        nx.relabel_nodes(self.graph, {old_node.id: replacement_node.id}, copy=False)
        self.graph.nodes[replacement_node.id]['label'] = replacement_node.get_label()

    def remove_chain(self, id_chain: list, heal=True, replace=False, new_chain_nodes: list = None):
        start_node = self.get_a_middle_node(node_id=id_chain[0])
        end_node = self.get_a_middle_node(node_id=id_chain[-1])
        start_parents = self.get_parents(start_node)
        end_children = self.get_children(end_node)

        for node_id in id_chain:
            self.all_nodes.pop(node_id)
            self.graph.remove_nodes_from(id_chain)

        if heal:
            for parent in start_parents:
                for child in end_children:
                    self.graph.add_edge(parent.id, child.id)

        if replace:
            current_parents = start_parents
            for node in new_chain_nodes:
                self.register_node(node)
                for parent in current_parents:
                    self.graph.add_edge(parent.id, node.id)
                current_parents = [node]

            for child in end_children:
                self.graph.add_edge(new_chain_nodes[-1].id, child.id)

    def get_chain_ids(self, start_id: int) -> list:
        node_ids = []
        current_node_id = start_id

        while True:
            successors = list(self.graph.successors(current_node_id))
            if len(successors) != 1:
                break
            node_ids.append(current_node_id)
            current_node_id = successors[0]

        return node_ids

    def mutate_node(self, position: int):
        nodes = self.get_mutatable_nodes()
        if position > (len(nodes) - 1):
            raise ValueError("Invalid request to mutate node: Length = " +
                             str(len(nodes)) + " position given as: " + str(position))

        node_to_mutate = list(nodes.values())[position]
        node_to_mutate.mutate()

    def get_output_nodes(self) -> dict:
        return {k: v for (k, v) in self.all_nodes.items() if k in self.output_ids}

    def get_input_nodes(self) -> dict:
        return {k: v for (k, v) in self.all_nodes.items() if k in self.input_ids}

    def get_not_input_nodes(self) -> dict:
        return {k: v for (k, v) in self.all_nodes.items() if k not in self.input_ids}

    def get_middle_nodes(self) -> dict:
        return {k: v for (k, v) in self.all_nodes.items()
                if (k not in self.input_ids) and (k not in self.output_ids)}

    def get_mutatable_nodes(self) -> dict:
        return {k: v for (k, v) in self.all_nodes.items() if v.can_mutate}

    def get_cx_nodes(self) -> dict:
        return {k: v for (k, v) in self.all_nodes.items() if
                (len(list(self.graph.predecessors(k))) == 1) and
                (len(list(self.graph.successors(k))) == 1)}

    def get_nodes_from_ids(self, id_list: list) -> dict:
        return {k: v for (k, v) in self.all_nodes.items() if k in id_list}

    def get_parents(self, node) -> list:
        parents_ids = self.graph.predecessors(node.id)  # iterator
        parents = []
        for p_id in parents_ids:
            parents.append(self.all_nodes[p_id])
        return parents

    def get_children(self, node) -> list:
        child_ids = self.graph.successors(node.id)
        children = []
        for c_id in child_ids:
            children.append(self.all_nodes[c_id])
        return children

    def get_a_middle_node(self, position=None, node_id=None) -> tensor_node.TensorNode:
        if (position is None) and (node_id is None):
            raise ValueError("Must specify either position or node_id")

        nodes = self.get_middle_nodes()
        node_selected = None
        if position is not None:
            if position > (len(nodes) - 1):
                raise ValueError("Invalid request to get node: Length = " +
                                 str(len(nodes)) + " position given as: " + str(position))
            node_selected = list(nodes.values())[position]
        else:
            try:
                node_selected = nodes[node_id]
            except KeyError:
                print("Invalid request to get node_id: " + str(node_id))
        return node_selected

    def __len__(self):
        return len(self.all_nodes)

    def draw_graphviz(self):
        py_graph = nx.nx_agraph.to_agraph(self.graph)
        py_graph.layout('dot')
        py_graph.draw('tensor_net_' + str(self.net_id) + '.png')

    def plot(self):
        pos = nx.nx_pydot.graphviz_layout(self.graph)
        nx.draw_networkx(self.graph, pos)
        plt.show()


def cx_single_node(tn: TensorNetwork, other_tn: TensorNetwork):
    tn_cx_nodes = tn.get_cx_nodes()
    other_cx_nodes = other_tn.get_cx_nodes()

    if len(tn_cx_nodes) == 0 or len(other_cx_nodes) == 0:
        return

    tn_node_id = random.choice(list(tn_cx_nodes.keys()))
    other_node_id = random.choice(list(other_cx_nodes.keys()))

    tn.replace_node(other_cx_nodes[other_node_id], node_id=tn_node_id)
    other_tn.replace_node(tn_cx_nodes[tn_node_id], node_id=other_node_id)


def cx_chain(tn: TensorNetwork, other_tn: TensorNetwork):
    tn_cx_nodes = tn.get_cx_nodes()
    other_cx_nodes = other_tn.get_cx_nodes()

    if len(tn_cx_nodes) == 0 or len(other_cx_nodes) == 0:
        return

    tn_node_id = random.choice(list(tn_cx_nodes.keys()))
    other_node_id = random.choice(list(other_cx_nodes.keys()))

    tn_chain_ids = tn.get_chain_ids(tn_node_id)
    tn_nodes = list(tn.get_nodes_from_ids(tn_chain_ids).values())
    other_chain_ids = other_tn.get_chain_ids(other_node_id)
    other_chain_nodes = list(other_tn.get_nodes_from_ids(other_chain_ids).values())

    tn.remove_chain(tn_chain_ids, heal=False, replace=True, new_chain_nodes=other_chain_nodes)
    other_tn.remove_chain(other_chain_ids, heal=False, replace=True, new_chain_nodes=tn_nodes)
