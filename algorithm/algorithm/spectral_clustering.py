import numpy as np
from scipy.sparse import linalg
from sklearn.cluster import KMeans


def get_spectral_input_from_cytoscape_json(data, drop_solo_nodes=True):
    """ data - dict - elements JSON for cytoscape
        drop_solo_nodes - bool - forget about nodes that have no connections

        all edges assumed to be bi-directional. Multiple edges between the same pair of nodes only counted as one adjacency

        returns a 3-tuple of
          degrees - list<int> - the degrees of the corresponding node in node_ids
          adjacencies - numpy.array (2-d) - has a 1 if nodes in the matrix are connected, 0 otherwise. Symmetric
          node_ids - list<str> - acts as a look up table of index to node_id
    """
    edges = data['edges']
    source_edge_node_set = {x['data']['source'] for x in edges}
    target_edge_node_set = {x['data']['target'] for x in edges}
    edge_node_set = source_edge_node_set | target_edge_node_set
    node_ids = [x['data']['id'] for x in data['nodes']]
    if drop_solo_nodes:  # drop nodes without edges
        node_ids = [x for x in node_ids if x in edge_node_set]
    n_node = len(node_ids)
    degrees = np.zeros(n_node, dtype=np.int8)
    adjacencies = np.zeros((n_node, n_node), dtype=np.int8)
    for i_node, i_id in enumerate(node_ids):
        edges_with_node_i = [x for x in edges
                             if x['data']['source'] == i_id or x['data']['target'] == i_id]
        nodes_adjacent_to_node_i = {x['data']['source'] for x in edges_with_node_i}
        nodes_adjacent_to_node_i |= {x['data']['target'] for x in edges_with_node_i}
        nodes_adjacent_to_node_i -= {i_id}  # omit self references
        degrees[i_node] = len(nodes_adjacent_to_node_i)
        for j_id in nodes_adjacent_to_node_i:
            j_node = node_ids.index(j_id)
            adjacencies[i_node, j_node] = adjacencies[j_node, i_node] = 1

    return degrees, adjacencies, node_ids


class SpectralClustererError(ValueError):
    pass


class SpectralClusterer:
    """ methods based on https://arxiv.org/abs/0711.0189 """
    methods = {'rw', 'sym', 'unnorm'}

    def __init__(self, adjacencies, node_ids, method='rw'):
        """ adjacencies - NxN symmetric matrix - relation/similarity of each node from the other
            node_ids - list<str> - length N list of labels/ids to associate to node indices in the adjacencies matrix
            method - str - from {'rw', 'sym', 'unnorm'}. See https://arxiv.org/abs/0711.0189 for details
        """
        if method not in self.methods:
            raise SpectralClustererError(f'encountered unknown method: {method}')
        self.adjacencies = adjacencies
        self.node_ids = node_ids
        self.method = method
        self.laplacian = self._get_laplacian(adjacencies, method)

    def _get_laplacian(self, adjacencies, method):
        """ returns an NxN matrix representing the Graph Laplacian (based on the requested method)
        """
        degrees = np.sum(adjacencies, axis=1).astype(np.float)
        laplacian = np.diag(degrees) - adjacencies
        if method == 'sym':
            m = np.diag(1.0 / np.sqrt(degrees))
            laplacian = np.matmul(m, laplacian)
            laplacian = np.matmul(laplacian, m)
        elif method == 'rw':
            m = np.diag(1.0 / np.array(degrees))
            laplacian = np.matmul(m, laplacian)
        return laplacian

    def get_eigvals(self):
        """ returns a list of eigen values of the graph Laplacian """
        return np.linalg.eigvals(self.laplacian)

    def get_eigvecs(self, n=None, orient=True):
        """ n - int - number of eigenvectors/values to return (ordered by eigenvalue from smallest)
            orient - bool - ensure first non-zero entry in eigenvector is positive
            returns a 2-tuple of
              - a list of eigenvalues
              - a list of the corresponding eigenvectors
        """
        if n is None:
            vals, vecs = np.linalg.eig(self.laplacian)
        else:
            cmplx_vals, cmplx_vecs = linalg.eigs(self.laplacian, k=n, which='SM')
            vecs = np.array([x.real for x in cmplx_vecs])
            vals = np.array([x.real for x in cmplx_vals])
        vecs = vecs[:, np.argsort(vals)]
        vals = vals[np.argsort(vals)]
        if orient:  # ensure first nonzero entry is positive
            vecs = np.transpose(vecs)
            for i_vec, vec in enumerate(vecs):
                nonzeroes = [x for x in vec if np.abs(x) > 1.0e-10]
                if nonzeroes:
                    if nonzeroes[0] < 0.0:
                        vecs[i_vec] = - vec
            vecs = np.transpose(vecs)
        return vals, vecs

    def get_node_vecs(self, start_evec, n_evec):
        """ start_evec - int - start eigenvector to use (indexed by eigenvalue, from lowest eigenvalue)
            n_evec - int - number of eigenvectors to use

            returns a list of feature vectors, corresponding to each node
        """
        _, eigvecs = self.get_eigvecs(n=start_evec + n_evec)
        eigvecs = eigvecs[:, start_evec:]
        if self.method == 'sym':
            for i_vec, vec in enumerate(eigvecs):
                eigvecs[i_vec] /= np.linalg.norm(vec)
        return eigvecs

    def get_kmeans_clusters(self, n_clusters, start_evec, n_evec):
        """ n_clusters - int - number of KMeans clusters
            start_evec - int
            n_evec - int

            returns a list<int> of cluster labels, corresponding to each node
        """
        node_vecs = self.get_node_vecs(start_evec, n_evec)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(node_vecs)
        return kmeans.labels_
