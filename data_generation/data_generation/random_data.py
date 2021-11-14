import numpy as np


def get_nodes(n_node, parent_weightings, seed=None):
    """ n_node - int - number of nodes to create
        parent_weightings - list<float> - length of list is the number of parents that 
            will be randomly associated to the nodes. The entries of the list are weightings 
            that form a probability distribution for drawing a parent index

        returns a list<dict> of JSON nodes in Cytoscape 'elements keyed by group' format
    """
    if seed is not None:
        np.random.seed(seed)
    parent_weightings = np.array(parent_weightings)
    parent_weightings = parent_weightings / np.sum(parent_weightings)
    parents = np.arange(len(parent_weightings))
    nodes = []
    for i_node in range(n_node):
        parent = np.random.choice(parents, p=parent_weightings)
        nodes.append({
            'data': {
                'id': f'node_{i_node}',
                'name': f'Node {i_node}',
                'parent': f'parent_{parent}'
            }
        })
    return nodes


def get_edges(nodes, in_parent_weight, out_of_parent_weight, min_edge_per_node, max_edge_per_node, seed=None):
    """ nodes - list - as returned by get_nodes
        in_parent_weight - float - weighting of the edge target to belong to the same parent as source
        out_of_parent_weight - float - weighting of the edge target to belong to a different parent to source
        min_edge_per_node - int - min number of outgoing edges from each source node
        max_edge_per_node - int - max(inclusive) number of outgoing edges from each source node

        the actual number of edges will be randomly chosen

        returns a list<dict> of JSON edges in Cytoscape 'elements keyed by group' format
    """
    if seed is not None:
        np.random.seed(0)
    parent_node_dict = {}
    for node in nodes:
        parent = node['data']['parent']
        node_list = parent_node_dict.get(parent, [])
        node_list.append(node['data']['id'])
        parent_node_dict[parent] = node_list
    n_edges_array = list(np.arange(min_edge_per_node, max_edge_per_node + 1))
    edges = []
    if n_edges_array:
        for node in nodes:
            parent = node['data']['parent']
            n_edges = np.random.choice(n_edges_array)
            in_parent_nodes = [x for x in parent_node_dict[parent] if x != node['data']['id']]
            out_of_parent_nodes = sum([v for k, v in parent_node_dict.items() if k != parent], [])
            this_targets = set()
            for i_edge in range(n_edges):
                is_in_parent_edge = np.random.random() < in_parent_weight/(in_parent_weight + out_of_parent_weight)
                if is_in_parent_edge and in_parent_nodes:
                    target = np.random.choice(in_parent_nodes)
                else:
                    target = np.random.choice(out_of_parent_nodes)
                this_targets.add(target)
            source = node['data']['id']
            this_edges = [{'data': {'id': f'{source}-{x}', 'source': source, 'target': x}} for x in this_targets]
            edges += this_edges
    return edges


def get_data(n_node, parent_weightings, in_parent_weight, out_of_parent_weight, min_edge_per_node, max_edge_per_node, seed=None):
    """ wrapper for get_nodes and get_edges

        returns an 'elements' dict in Cytoscape 'elements keyed by group' format
    """
    if seed is not None:
        np.random.seed(seed)
    nodes = get_nodes(n_node, parent_weightings)
    edges = get_edges(nodes, in_parent_weight, out_of_parent_weight, min_edge_per_node, max_edge_per_node)
    return {'nodes': nodes, 'edges': edges}
