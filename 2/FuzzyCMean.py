import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

class FuzzyCMean:
    def __init__(self, k=5, m=10, number_of_iterations=100, file_name='data1.csv'):
        self.file_name = file_name
        self.data = np.genfromtxt(self.file_name, delimiter=',')
        self.m = m
        self.k = k
        self.u = None
        self.centroids = None
        self.number_of_iterations = number_of_iterations
        self.error = float('inf')
        self.colors = ['r', 'y', 'g', 'c', 'b', 'm']
        # plt.style.use('dark_background')

    def scatter_plot(self):
        plt.figure(figsize=(8, 5))
        plt.scatter(self.data[:, 0], self.data[:, 1])
        for k in range(self.k):
            cluster = np.array([
                self.data[i]
                for i in range(len(self.data))
                if k == np.argmax(self.u[i])
            ])
            plt.scatter(cluster[:, 0], cluster[:, 1], alpha=0.7, c=self.colors[k], s=50)

        for centroid in range(len(self.centroids)):
            plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1], color=self.colors[centroid],
                        edgecolors='k', linewidths=2, s=100,
                        label='centroid #{}\n({}, {})'.format(centroid+1,
                                                              round(self.centroids[centroid][0], 2),
                                                              round(self.centroids[centroid][1], 2)))
            plt.axvline(x=self.centroids[centroid][0], c=self.colors[centroid], linestyle='dashed', linewidth=0.3)
            plt.axhline(y=self.centroids[centroid][1], c=self.colors[centroid], linestyle='dashed', linewidth=0.3)

        plt.title('k: {}, m: {}'.format(self.k, self.m))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def clustering(self):
        self.centroids = np.array([
            np.array([random.uniform(min(self.data[:, 0]), max(self.data[:, 1])) for d in range(self.data.shape[1])])
            for k in range(self.k)])

        for i in tqdm(range(self.number_of_iterations), desc='m = {},\tk = {}'.format(self.m, self.k)):
            self.u = np.array([
                [
                    1 / np.sum(
                        [(np.linalg.norm(Xk - Vi) / np.linalg.norm(Xk - Vj)) ** (2 / (self.m - 1))
                         for Vj in self.centroids]
                    )
                    for Vi in self.centroids
                ]
                for Xk in self.data
            ])

            self.centroids = np.array([
                np.add.reduce([
                    (self.u[kx, i] ** self.m) * self.data[kx] for kx in range(len(self.data))])
                / np.sum([sample[i] ** self.m for sample in self.u]) for i in range(len(self.centroids))])

            self.error = np.sum([
                np.sum([(self.u[j, i] ** self.m) * (np.linalg.norm(self.data[j] - self.centroids[i]) ** 2)
                        for i in range(len(self.centroids))]) for j in range(len(self.data))])


errors = []
for i in range(1, 6):
    model = FuzzyCMean(k=i, file_name='data1.csv')
    model.clustering()
    model.scatter_plot()
    errors.append(model.error)


plt.plot(np.arange(1, 6), errors)
plt.title('Errors')
plt.xlabel('# CLUSTERS')
plt.ylabel('COST')
plt.show()

