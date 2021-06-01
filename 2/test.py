import csv
import matplotlib.pyplot as plt
import numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")


class FuzzyCMean:
    def __init__(self):
        with open('data1.csv', 'r') as csv_file:
            # creating a csv reader object
            csv_reader = csv.reader(csv_file)

            # extracting each data row one by one
            rows = []
            for row in csv_reader:
                try:
                    rows.append(list([float(row[0]), float(row[1])]))
                except:
                    pass
        self.data = np.array(rows)
        self.k = 2
        self.m = 2

        # plotting
        self.colors = ['r', 'y', 'g', 'c', 'b', 'm']
        plt.style.use('dark_background')

        # creating clusters
        self.clusters = []
        for k in range(self.k):
            self.clusters.append(np.array([]))

        # initializing k centroids randomly
        self.centroids = self.data[np.random.choice(self.data.shape[0], self.k, replace=False), :]

    def clear_clusters(self):
        for k in range(self.k):
            self.clusters[k] = np.array([])

    def clustering(self):
        """
        a method to find out each data is for which cluster by looking at centroids
        """
        for d in self.data:
            u = []
            for c in self.centroids:
                uu = self.u(d, c, self.m)
                u.append(uu)
            cluster = int(np.argmax(u))
            self.clusters[cluster] = np.append(self.clusters[cluster], d)

    def update_centroids(self):
        pass

    def u(self, Xk, Vi, m):
        """
        a method to determine how much data Xk is belonged to cluster Vi

        :param Xk: data
        :param Vi: target cluster
        :param m: a parameter (greater than 1)
        :return: u
        """
        Xk_Vi = np.linalg.norm(Xk - Vi)
        s = 0
        for Vj in self.centroids:
            s += (Xk_Vi / np.linalg.norm(Xk - Vj))**(2 / (m - 1))
        return 1 / s

    def plotting_clusters(self):
        """
        a method to plot clusters in a scatter diagram (2D)
        :return:
        """
        for cluster, color in zip(self.clusters, self.colors[:self.k]):
            for data_of_cluster in cluster.reshape((-1, 2)):
                plt.scatter(data_of_cluster[0], data_of_cluster[1], c=color, alpha=0.5)

    def main(self):
        for i in tqdm(range(100)):
            self.clear_clusters()
            self.clustering()
            self.update_centroids()

        self.plotting_clusters()
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=100, c='w', edgecolors='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()









# print(data)


if __name__ == '__main__':
    FuzzyCMean().main()



