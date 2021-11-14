from pytest import raises
import numpy as np
from ..algorithm.spectral_clustering import get_spectral_input_from_cytoscape_json
from ..algorithm.spectral_clustering import SpectralClusterer
from ..algorithm.spectral_clustering import SpectralClustererError

node_ids = ['n0', 'n1', 'n2', 'n3', 'n4']
adjacencies = [[0, 1, 1, 1, 0],
               [1, 0, 0, 0, 0],
               [1, 0, 0, 0, 1],
               [1, 0, 0, 0, 0],
               [0, 0, 1, 0, 0]]


def test_degrees_and_adjacencies_from_cytoscape_json():
    nodes = [{'data': {'id': 'n0', 'name': 'n 0', 'parent': 'p0'}},
             {'data': {'id': 'n1', 'name': 'n 1', 'parent': 'p0'}},
             {'data': {'id': 'n2', 'name': 'n 2', 'parent': 'p0'}},
             {'data': {'id': 'n3', 'name': 'n 3', 'parent': 'p1'}},
             {'data': {'id': 'n4', 'name': 'n 4', 'parent': 'p1'}},
             {'data': {'id': 'n5', 'name': 'n 5', 'parent': 'p1'}},
             {'data': {'id': 'n6', 'name': 'n 6', 'parent': 'p2'}},
             {'data': {'id': 'n7', 'name': 'n 7', 'parent': 'p2'}},
             {'data': {'id': 'n8', 'name': 'n 8', 'parent': 'p2'}}]
    edges = [{'data': {'id': 'n0-n5', 'source': 'n0', 'target': 'n5'}},
             {'data': {'id': 'n1-n6', 'source': 'n1', 'target': 'n6'}},
             {'data': {'id': 'n4-n3', 'source': 'n4', 'target': 'n3'}},
             {'data': {'id': 'n5-n0', 'source': 'n5', 'target': 'n0'}},  # deliberate reversal of another edge
             {'data': {'id': 'n5-n8', 'source': 'n5', 'target': 'n8'}},
             {'data': {'id': 'n5-n8', 'source': 'n5', 'target': 'n8'}},  # deliberate duplicate
             {'data': {'id': 'n6-n0', 'source': 'n6', 'target': 'n0'}},
             {'data': {'id': 'n7-n0', 'source': 'n7', 'target': 'n0'}},
             {'data': {'id': 'n8-n7', 'source': 'n8', 'target': 'n7'}}]
    data = {'nodes': nodes, 'edges': edges}

    degrees, adjacencies, node_ids = get_spectral_input_from_cytoscape_json(data, drop_solo_nodes=True)
    assert node_ids == ['n0', 'n1', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']
    assert list(degrees) == [3, 1, 1, 1, 2, 2, 2, 2]
    np.testing.assert_array_equal(adjacencies, [[0, 0, 0, 0, 1, 1, 1, 0],
                                                [0, 0, 0, 0, 0, 1, 0, 0],
                                                [0, 0, 0, 1, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 0, 0, 0, 1],
                                                [1, 1, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 1, 0, 1, 0]])

    degrees, adjacencies, node_ids = get_spectral_input_from_cytoscape_json(data, drop_solo_nodes=False)
    assert node_ids == ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']
    assert list(degrees) == [3, 1, 0, 1, 1, 2, 2, 2, 2]

    np.testing.assert_array_equal(adjacencies, [[0, 0, 0, 0, 0, 1, 1, 1, 0],
                                                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                                [1, 1, 0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 1, 0, 1, 0]])


def test_spectral_clusterer_init():
    s = SpectralClusterer(adjacencies, node_ids)
    assert s.method == 'rw'
    with raises(SpectralClustererError):
        SpectralClusterer(adjacencies, node_ids, method='i_am_not_a_method')


def test_spectral_clusterer_rw():
    s = SpectralClusterer(adjacencies, node_ids, method='rw')
    assert s.method == 'rw'
    np.testing.assert_allclose(s.laplacian, [[1.0, -0.33333333, -0.33333333, -0.33333333,  0.0],
                                             [-1.0, 1.0, 0.0, 0.0, 0.0],
                                             [-0.5, 0.0, 1.0, 0.0, -0.5],
                                             [-1.0, 0.0, 0.0, 1.0, 0.0],
                                             [0.0, 0.0, -1.0, 0.0, 1.0]])
    np.testing.assert_allclose(s.get_eigvals(), [0.0, 4.22649731e-01, 2.0, 1.57735027, 1.0], atol=1.0e-15)
    vals, vecs = s.get_eigvecs()
    np.testing.assert_allclose(vals, [0.0, 4.22649731e-01, 1.0, 1.57735027, 2.0], atol=1.0e-15)
    np.testing.assert_allclose(vecs, [[4.47213595e-01,  2.29415734e-01, 0.0,              2.29415734e-01, 4.47213595e-01],
                                      [4.47213595e-01,  3.97359707e-01, 7.07106781e-01,  -3.97359707e-01, -4.47213595e-01],
                                      [4.47213595e-01, -3.97359707e-01, 0.0,              3.97359707e-01, -4.47213595e-01],
                                      [4.47213595e-01,  3.97359707e-01, -7.07106781e-01, -3.97359707e-01, -4.47213595e-01],
                                      [4.47213595e-01, -6.88247202e-01, 0.0,             -6.88247202e-01, 4.47213595e-01]], atol=1.0e-7)
    vals, vecs = s.get_eigvecs(n=2)
    np.testing.assert_allclose(vals, [0.0, 4.22649731e-01], atol=1.0e-15)
    np.testing.assert_allclose(vecs, [[4.47213595e-01,  2.29415734e-01],
                                      [4.47213595e-01,  3.97359707e-01],
                                      [4.47213595e-01, -3.97359707e-01],
                                      [4.47213595e-01,  3.97359707e-01],
                                      [4.47213595e-01, -6.88247202e-01]], atol=1.0e-7)
    node_vecs = s.get_node_vecs(1, 2)
    np.testing.assert_allclose(node_vecs, [[2.29415734e-01, 0.0],
                                           [3.97359707e-01, 7.07106781e-01],
                                           [-3.97359707e-01, 0.0],
                                           [3.97359707e-01, -7.07106781e-01],
                                           [-6.88247202e-01, 0.0]], atol=1.0e-7)

    k = list(s.get_kmeans_clusters(2, 1, 1))
    assert k == [1, 1, 0, 1, 0] or k == [0, 0, 1, 0, 1]


def test_spectral_clusterer_unnorm():
    s = SpectralClusterer(adjacencies, node_ids, method='unnorm')
    assert s.method == 'unnorm'
    np.testing.assert_allclose(s.laplacian, [[3, -1, -1, -1,  0],
                                             [-1,  1,  0,  0,  0],
                                             [-1,  0,  2,  0, -1],
                                             [-1,  0,  0,  1,  0],
                                             [0,  0, -1,  0,  1]])
    np.testing.assert_allclose(s.get_eigvals(), [4.170086, 2.311108, 0.0, 5.188057e-01, 1.0], atol=1.0e-7)
    vals, vecs = s.get_eigvecs()
    np.testing.assert_allclose(vals, [0.0, 5.188057e-01, 1.0, 2.311108, 4.170086], atol=1.0e-7)
    np.testing.assert_allclose(vecs, [[4.47213595e-01,  2.01774144e-01, 0.0, 3.17515788e-01, 8.11462211e-01],
                                      [4.47213595e-01,  4.19319477e-01, 7.07106781e-01, -2.42173667e-01, -2.55974786e-01],
                                      [4.47213595e-01, -3.37998097e-01, 0.0, 7.03081478e-01, -4.37531395e-01],
                                      [4.47213595e-01,  4.19319477e-01, -7.07106781e-01, -2.42173667e-01, -2.55974786e-01],
                                      [4.47213595e-01, -7.02415001e-01, 0.0, -5.36249932e-01, 1.38018756e-01]], atol=1.0e-7)

    vals, vecs = s.get_eigvecs(n=2)
    np.testing.assert_allclose(vals, [0.0, 5.188057e-01], atol=1.0e-7)
    np.testing.assert_allclose(vecs, [[4.47213595e-01,  2.01774144e-01],
                                      [4.47213595e-01,  4.19319477e-01],
                                      [4.47213595e-01, -3.37998097e-01],
                                      [4.47213595e-01,  4.19319477e-01],
                                      [4.47213595e-01, -7.02415001e-01]], atol=1.0e-7)

    node_vecs = s.get_node_vecs(1, 2)
    np.testing.assert_allclose(node_vecs, [[2.01774144e-01, 0.0],
                                           [4.19319477e-01, 7.07106781e-01],
                                           [-3.37998097e-01, 0.0],
                                           [4.19319477e-01, -7.07106781e-01],
                                           [-7.02415001e-01, 0.0]], atol=1.0e-7)

    k = list(s.get_kmeans_clusters(3, 1, 2))  # classes are nodes [0,3], [1], [2,4]
    assert k[0] == k[3]
    assert k[2] == k[4]
    assert k[1] != k[0]
    assert k[1] != k[2]
    assert k[0] != k[2]


def test_spectral_clusterer_sym():
    s = SpectralClusterer(adjacencies, node_ids, method='sym')
    assert s.method == 'sym'
    np.testing.assert_allclose(s.laplacian, [[1.0, -0.57735027, -0.40824829, -0.57735027, 0.0],
                                             [-0.57735027, 1.0, 0.0, 0.0, 0.0],
                                             [-0.40824829, 0.0, 1.0, 0.0, -0.70710678],
                                             [-0.57735027, 0.0, 0.0, 1.0, 0.0],
                                             [0.0, 0.0, -0.70710678, 0.0, 1.0]])
    np.testing.assert_allclose(s.get_eigvals(), [0.0, 4.22649731e-01, 2.0, 1.57735027, 1.0], atol=1.0e-15)
    vals, vecs = s.get_eigvecs()
    np.testing.assert_allclose(vals, [0.0, 4.22649731e-01, 1.0, 1.57735027, 2.0], atol=1.0e-7)
    np.testing.assert_allclose(vecs, [[6.12372436e-01, 3.53553391e-01, 0.0,  3.53553391e-01, 6.12372436e-01],
                                      [3.53553391e-01, 3.53553391e-01, 7.07106781e-01, -3.53553391e-01, -3.53553391e-01],
                                      [5.00000000e-01, -5.00000000e-01, 0.0, 5.00000000e-01, -5.00000000e-01],
                                      [3.53553391e-01, 3.53553391e-01, -7.07106781e-01, -3.53553391e-01, -3.53553391e-01],
                                      [3.53553391e-01, -6.12372436e-01, 0.0, -6.12372436e-01, 3.53553391e-01]], atol=1.0e-7)

    vals, vecs = s.get_eigvecs(n=3)
    np.testing.assert_allclose(vals, [0.0, 4.22649731e-01, 1.0], atol=1.0e-7)
    np.testing.assert_allclose(vecs, [[6.12372436e-01, 3.53553391e-01, 0.0],
                                      [3.53553391e-01, 3.53553391e-01, 7.07106781e-01],
                                      [5.00000000e-01, -5.00000000e-01, 0.0],
                                      [3.53553391e-01, 3.53553391e-01, -7.07106781e-01],
                                      [3.53553391e-01, -6.12372436e-01, 0.0]], atol=1.0e-7)

    node_vecs = s.get_node_vecs(1, 2)
    np.testing.assert_allclose(node_vecs, [[1.0, 0.0],
                                           [4.47213595e-01,  8.94427191e-01],
                                           [-1.0,  0.0],
                                           [4.47213595e-01, -8.94427191e-01],
                                           [-1.0,  0.0]], atol=1.0e-7)

    k = list(s.get_kmeans_clusters(2, 1, 1))
    assert k == [1, 1, 0, 1, 0] or k == [0, 0, 1, 0, 1]
