import numpy as np
from sklearn.neighbors import kneighbors_graph


def process_graph(adj):
    A = np.asmatrix(adj)
    I = np.eye(len(A))
    A_hat = A + I
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = D_hat**0.5
    D_hat = np.matrix(np.diag(D_hat))
    D_hat = D_hat**-1
    return A_hat, D_hat


def distance(features_1, features_2):
    dist = np.sqrt(np.sum(np.square(features_1 - features_2)))
    return dist


def supervised_graph(features_label, labels):
    adj_labels = np.zeros((len(features_label), len(features_label)))
    for i in range(len(features_label)):
        for j in range(len(features_label)):
            if labels[i] == labels[j]:
                adj_labels[i, j] = 1
    return adj_labels


def from_labels_get_radius(features, labels, class_number):
    distance_list = np.zeros(class_number)
    for index1, feature1 in enumerate(features):
        for index2, feature2 in enumerate(features):
            if labels[index1] == labels[index2]:
                distance_list[labels[index1]] += distance(features[index1], features[index2])
    count = (len(features) * (len(features) - 1)) / 2
    radius_limit = np.mean(distance_list / count)
    return radius_limit


def radius_graph(features, dis, labels=None, class_number=None, include_self=True, limit_radius=False):
    if limit_radius:
        dis = from_labels_get_radius(features, labels, class_number)
    adjacency_matrix = np.zeros((len(features), len(features)))
    for index1, feature1 in enumerate(features):
        for index2, feature2 in enumerate(features):
            if dis >= distance(feature1, feature2) and index1 != index2:
                adjacency_matrix[index1, index2] = 1
            elif dis >= distance(feature1, feature2) and index1 == index2 and include_self:
                adjacency_matrix[index1, index2] = 1
    return adjacency_matrix


def knn_and_radius_graph(features, k, dis=None, labels=None, class_number=None, include_self=True, limit_radius=False):
    adjacency_matrix_knn = kneighbors_graph(features, k, include_self=include_self).toarray()
    if limit_radius:
        features_label = features[:len(labels)]
        dis = from_labels_get_radius(features_label, labels, class_number)
    adjacency_matrix_radius = radius_graph(features, dis, include_self)
    adjacency_matrix = np.zeros((len(features), len(features)))
    for index in range(len(features)):
        radius_node_number = np.sum(adjacency_matrix_radius[index])
        if radius_node_number > k:
            adjacency_matrix[index] = adjacency_matrix_radius[index]
        else:
            adjacency_matrix[index] = adjacency_matrix_knn[index]
    if limit_radius:
        return adjacency_matrix, dis
    else:
        return adjacency_matrix


def alter_knn_and_radius_graph1(features, k, labels, dis=None, class_number=None, include_self=True, limit_radius=False):
    features_label = features[:len(labels)]
    adj_supervised = supervised_graph(features_label, labels)
    if limit_radius:
        adj_all, dis = knn_and_radius_graph(features, k, dis, labels, class_number, include_self, limit_radius)
        adj_all[:len(adj_supervised), :len(adj_supervised)] = adj_supervised
        return adj_all, dis
    else:
        adj_all = knn_and_radius_graph(features, k, dis, labels, class_number, include_self, limit_radius)
        adj_all[:len(adj_supervised), :len(adj_supervised)] = adj_supervised
        return adj_all


if __name__ == '__main__':
    print(np.ones(2))




