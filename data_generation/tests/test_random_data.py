from ..data_generation.random_data import get_nodes
from ..data_generation.random_data import get_edges
from ..data_generation.random_data import get_data


def test_get_nodes():
    nodes = get_nodes(4, [1], seed=0)
    assert nodes == [{'data': {'id': 'node_0', 'name': 'Node 0', 'parent': 'parent_0'}},
                     {'data': {'id': 'node_1', 'name': 'Node 1', 'parent': 'parent_0'}},
                     {'data': {'id': 'node_2', 'name': 'Node 2', 'parent': 'parent_0'}},
                     {'data': {'id': 'node_3', 'name': 'Node 3', 'parent': 'parent_0'}}]

    nodes = get_nodes(4, [1, 1], seed=1)
    assert nodes == [{'data': {'id': 'node_0', 'name': 'Node 0', 'parent': 'parent_0'}},
                     {'data': {'id': 'node_1', 'name': 'Node 1', 'parent': 'parent_1'}},
                     {'data': {'id': 'node_2', 'name': 'Node 2', 'parent': 'parent_0'}},
                     {'data': {'id': 'node_3', 'name': 'Node 3', 'parent': 'parent_0'}}]

    nodes = get_nodes(9, [0.1, 0.2, 0.3], seed=999)  # mostly parent 2s in data
    assert nodes == [{'data': {'id': 'node_0', 'name': 'Node 0', 'parent': 'parent_2'}},
                     {'data': {'id': 'node_1', 'name': 'Node 1', 'parent': 'parent_2'}},
                     {'data': {'id': 'node_2', 'name': 'Node 2', 'parent': 'parent_0'}},
                     {'data': {'id': 'node_3', 'name': 'Node 3', 'parent': 'parent_2'}},
                     {'data': {'id': 'node_4', 'name': 'Node 4', 'parent': 'parent_0'}},
                     {'data': {'id': 'node_5', 'name': 'Node 5', 'parent': 'parent_1'}},
                     {'data': {'id': 'node_6', 'name': 'Node 6', 'parent': 'parent_1'}},
                     {'data': {'id': 'node_7', 'name': 'Node 7', 'parent': 'parent_2'}},
                     {'data': {'id': 'node_8', 'name': 'Node 8', 'parent': 'parent_2'}}]


def test_get_edges():
    nodes = [{'data': {'id': 'n0', 'name': 'n 0', 'parent': 'p0'}},
             {'data': {'id': 'n1', 'name': 'n 1', 'parent': 'p0'}},
             {'data': {'id': 'n2', 'name': 'n 2', 'parent': 'p0'}},
             {'data': {'id': 'n3', 'name': 'n 3', 'parent': 'p1'}},
             {'data': {'id': 'n4', 'name': 'n 4', 'parent': 'p1'}},
             {'data': {'id': 'n5', 'name': 'n 5', 'parent': 'p1'}},
             {'data': {'id': 'n6', 'name': 'n 6', 'parent': 'p2'}},
             {'data': {'id': 'n7', 'name': 'n 7', 'parent': 'p2'}},
             {'data': {'id': 'n8', 'name': 'n 8', 'parent': 'p2'}}]

    edges = get_edges(nodes, 1.0, 1.0, 1, 1, seed=0)  # mix of in parent and out of parent edges
    assert edges == [{'data': {'id': 'n0-n8', 'source': 'n0', 'target': 'n8'}},
                     {'data': {'id': 'n1-n6', 'source': 'n1', 'target': 'n6'}},
                     {'data': {'id': 'n2-n4', 'source': 'n2', 'target': 'n4'}},
                     {'data': {'id': 'n3-n2', 'source': 'n3', 'target': 'n2'}},
                     {'data': {'id': 'n4-n3', 'source': 'n4', 'target': 'n3'}},
                     {'data': {'id': 'n5-n3', 'source': 'n5', 'target': 'n3'}},
                     {'data': {'id': 'n6-n7', 'source': 'n6', 'target': 'n7'}},
                     {'data': {'id': 'n7-n0', 'source': 'n7', 'target': 'n0'}},
                     {'data': {'id': 'n8-n7', 'source': 'n8', 'target': 'n7'}}]

    edges = get_edges(nodes, 1.0, 0.0, 1, 1, seed=0)  # all in parent edges
    assert edges == [{'data': {'id': 'n0-n2', 'source': 'n0', 'target': 'n2'}},
                     {'data': {'id': 'n1-n2', 'source': 'n1', 'target': 'n2'}},
                     {'data': {'id': 'n2-n1', 'source': 'n2', 'target': 'n1'}},
                     {'data': {'id': 'n3-n4', 'source': 'n3', 'target': 'n4'}},
                     {'data': {'id': 'n4-n3', 'source': 'n4', 'target': 'n3'}},
                     {'data': {'id': 'n5-n3', 'source': 'n5', 'target': 'n3'}},
                     {'data': {'id': 'n6-n7', 'source': 'n6', 'target': 'n7'}},
                     {'data': {'id': 'n7-n6', 'source': 'n7', 'target': 'n6'}},
                     {'data': {'id': 'n8-n7', 'source': 'n8', 'target': 'n7'}}]

    edges = get_edges(nodes, 0.0, 1.0, 1, 1, seed=0)  # all out of parent edges
    assert edges == [{'data': {'id': 'n0-n8', 'source': 'n0', 'target': 'n8'}},
                     {'data': {'id': 'n1-n6', 'source': 'n1', 'target': 'n6'}},
                     {'data': {'id': 'n2-n4', 'source': 'n2', 'target': 'n4'}},
                     {'data': {'id': 'n3-n2', 'source': 'n3', 'target': 'n2'}},
                     {'data': {'id': 'n4-n0', 'source': 'n4', 'target': 'n0'}},
                     {'data': {'id': 'n5-n2', 'source': 'n5', 'target': 'n2'}},
                     {'data': {'id': 'n6-n0', 'source': 'n6', 'target': 'n0'}},
                     {'data': {'id': 'n7-n1', 'source': 'n7', 'target': 'n1'}},
                     {'data': {'id': 'n8-n1', 'source': 'n8', 'target': 'n1'}}]

    edges = get_edges(nodes, 1.0, 1.0, 0, 2, seed=0)  # 0-2 source edges per node
    edges = sorted(edges, key=lambda x: x['data']['id'])  # sort since will have multiple source nodes
    assert edges == [{'data': {'id': 'n1-n6', 'source': 'n1', 'target': 'n6'}},
                     {'data': {'id': 'n2-n5', 'source': 'n2', 'target': 'n5'}},
                     {'data': {'id': 'n4-n3', 'source': 'n4', 'target': 'n3'}},
                     {'data': {'id': 'n5-n0', 'source': 'n5', 'target': 'n0'}},
                     {'data': {'id': 'n5-n8', 'source': 'n5', 'target': 'n8'}},
                     {'data': {'id': 'n6-n0', 'source': 'n6', 'target': 'n0'}},
                     {'data': {'id': 'n7-n0', 'source': 'n7', 'target': 'n0'}},
                     {'data': {'id': 'n8-n7', 'source': 'n8', 'target': 'n7'}}]


def test_get_data():
    data = get_data(4, [1, 1], 1.0, 1.0, 1, 1, seed=0)
    assert data == {
        'nodes': [
            {'data': {'id': 'node_0', 'name': 'Node 0', 'parent': 'parent_1'}},
            {'data': {'id': 'node_1', 'name': 'Node 1', 'parent': 'parent_1'}},
            {'data': {'id': 'node_2', 'name': 'Node 2', 'parent': 'parent_1'}},
            {'data': {'id': 'node_3', 'name': 'Node 3', 'parent': 'parent_1'}}
        ],
        'edges': [
            {'data': {'id': 'node_0-node_2', 'source': 'node_0', 'target': 'node_2'}},
            {'data': {'id': 'node_1-node_3', 'source': 'node_1', 'target': 'node_3'}},
            {'data': {'id': 'node_2-node_0', 'source': 'node_2', 'target': 'node_0'}},
            {'data': {'id': 'node_3-node_2', 'source': 'node_3', 'target': 'node_2'}}
        ]
    }
