from math import pi, sin, cos
from random import choice, choices, gauss, uniform

import matplotlib.pyplot as plt

def calculate_centroid(data):
    data_x, data_y = zip(*data)
    return (sum(data_x) / len(data), sum(data_y) / len(data))

def distance_squared(a, b):
    return pow(abs(a[0] - b[0]), 2) + pow(abs(a[1] - b[1]), 2)

def cluster(data, centroids):
    clusters = [list() for _ in centroids]
    for data_point in data:
        centroid_distances_sqaured = [distance_squared(data_point, centroid) for centroid in centroids]
        cluster_index = min(range(len(centroids)), key=centroid_distances_sqaured.__getitem__)
        clusters[cluster_index].append(data_point)
    return clusters

def calculate_centroids(clusters):
    centroids = [calculate_centroid(cluster) for cluster in clusters]
    centroids.sort()
    return centroids

def choose_seeds(data, num_seeds):
    seeds = [choice(data)]
    while len(seeds) < num_seeds:
        weights = []
        for data_point in data:
            weights.append(min([distance_squared(data_point, seed) for seed in seeds]))
        seeds.extend(choices(population=data, weights=weights, k=1))
    seeds.sort()
    return seeds

def do_k_means_clustering(data, num_clusters):
    centroids = choose_seeds(data, num_clusters)

    prev_centroids = []
    clusters = []
    while set(centroids) != set(prev_centroids):
        prev_centroids = centroids

        clusters = cluster(data, centroids)

        centroids = calculate_centroids(clusters)

    return (centroids, clusters)

def generate_data_cluster(centroid_x, centroid_y, num_points, std_dev = 2):
    data_cluster = []
    for _ in range(num_points):
        dist = gauss(0, std_dev)
        arg = uniform(-pi, pi)
        data_cluster.append((centroid_x + dist * cos(arg), centroid_y + dist * sin(arg)))
    return data_cluster

def k_means_example():
    num_clusters = 3
    num_points_per_cluster = 200
    origins = [(5, 4), (8, 2), (11, 6)]

    data = []
    for (x, y) in origins:
        data.extend(generate_data_cluster(x, y, num_points_per_cluster))

    (centroids, clusters) = do_k_means_clustering(data, num_clusters)

    plt.style.use('ggplot')
    plt.figure(1)

    plt.subplot(211)
    data_x, data_y = zip(*data)
    plt.scatter(data_x, data_y, c='b')
    origins_x, origins_y = zip(*origins)
    plt.scatter(origins_x, origins_y, c='r')

    plt.subplot(212)
    for clstr in clusters:
        cluster_x, cluster_y = zip(*clstr)
        plt.scatter(cluster_x, cluster_y)
    centroids_x, centroids_y = zip(*centroids)
    plt.scatter(centroids_x, centroids_y)

    plt.show()

k_means_example()
