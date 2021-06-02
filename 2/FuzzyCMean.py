import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


class FuzzyCMean:
    def __init__(self, k=5, m=2, number_of_iterations=100, file_name='data1.csv'):
        self.file_name = file_name
        self.data = np.genfromtxt(self.file_name, delimiter=',')
        self.m = m
        self.k = k
        self.u = None
        self.centroids = None
        self.number_of_iterations = number_of_iterations
        self.error = float('inf')
        self.colors = ['r', 'y', 'g', 'c', 'b', 'm']

    def scatter_plot(self):
        """ a method to plot clusters in a scatter diagram (2D) """
        # set size for figure
        plt.figure(figsize=(8, 5))

        # plotting cluster elements
        for k in range(self.k):
            cluster = np.array([self.data[i] for i in range(len(self.data)) if k == np.argmax(self.u[i])])
            plt.scatter(cluster[:, 0], cluster[:, 1], alpha=0.7, c=self.colors[k], s=50)

        # plotting centroids
        for centroid in range(len(self.centroids)):
            plt.scatter(self.centroids[centroid][0],
                        self.centroids[centroid][1],
                        color=self.colors[centroid],
                        edgecolors='k', linewidths=2, s=100,
                        label='centroid #{}\n({}, {})'.format(centroid+1,
                                                              round(self.centroids[centroid][0], 2),
                                                              round(self.centroids[centroid][1], 2)))

        plt.title('k: {}, m: {}'.format(self.k, self.m))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def generate_initial_centroids(self):
        """ choosing k distinct data randomly as centroids """
        self.centroids = np.array([
            np.array([random.uniform(min(self.data[:, 0]), max(self.data[:, 1])) for d in range(self.data.shape[1])])
            for k in range(self.k)])

    def calculate_u(self):
        """ a method to determine how much data Xk is belonged to cluster Vi """
        self.u = np.array([[
                1 / np.sum(
                        [(np.linalg.norm(Xk - Vi) / np.linalg.norm(Xk - Vj)) ** (2 / (self.m - 1))for Vj in self.centroids])
                    for Vi in self.centroids]
                for Xk in self.data
            ])

    def update_centroids(self):
        """ updating centroids using data in the clusters """
        self.centroids = np.array([
                np.add.reduce([
                    (self.u[d, c] ** self.m) * self.data[d] for d in range(len(self.data))])
                / np.sum([sample[c] ** self.m for sample in self.u]) for c in range(len(self.centroids))])

    def calculate_errors(self):
        """ calculating error for the whole model """
        self.error = np.sum([
                np.sum([(self.u[j, i] ** self.m) * (np.linalg.norm(self.data[j] - self.centroids[i]) ** 2)
                        for i in range(len(self.centroids))]) for j in range(len(self.data))])

    def clustering(self):
        self.generate_initial_centroids()
        desc = 'dataset = {}, m = {}, k = {}'.format(self.file_name, self.m, self.k)
        for i in tqdm(range(self.number_of_iterations), desc=desc):
            self.calculate_u()
            self.update_centroids()
            self.calculate_errors()

k_range = 6
for i in range(4):
    errors = []
    for k in range(1, k_range+1):
        model = FuzzyCMean(k=k, file_name='data{}.csv'.format(i+1))
        model.clustering()
        errors.append(model.error)

    plt.plot(np.arange(1, k_range+1), errors, color='r')
    plt.title('Errors for \"data{}.csv\"'.format(i+1))
    plt.xlabel('# CLUSTERS')
    plt.ylabel('COST')
    plt.show()

M = 10
errors_for_m = []
for m in range(1, M):
    model = FuzzyCMean(k=2, file_name='data1.csv', m=m + 1)
    model.clustering()
    errors_for_m.append(model.error)

plt.plot(np.arange(1, M), errors_for_m, color='g')
plt.title('Errors for \"data{}.csv\" with different values for m'.format(1))
plt.xlabel('m')
plt.ylabel('COST')
plt.show()

for i in range(1, 6):
    f1 = FuzzyCMean(k=i, file_name='data1.csv')
    f1.clustering()
    f1.scatter_plot()

for i in range(1, 6):
    f1 = FuzzyCMean(k=i, file_name='data3.csv')
    f1.clustering()
    f1.scatter_plot()