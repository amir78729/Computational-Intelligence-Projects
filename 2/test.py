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
        self.u = None
        self.k = 3
        self.m = 2
        self.error = float('inf')
        self.number_of_iteration = 10

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

    def number_of_iterations(self):
        """
        a method to find out each data is for which cluster by looking at centroids
        """
        # self.clear_clusters()
        # for d in self.data:
        #     u = []
        #     for c in self.centroids:
        #         uu = self.u(d, c)
        #         u.append(uu)
        #     cluster = int(np.argmax(u))
        #     self.clusters[cluster] = np.append(self.clusters[cluster], d)

    def update_centroids(self):
        # for Vi in range(len(self.centroids)):
        #     numerator = 0
        #     denominator = 0
        #     for Xk in self.data:
        #         u = self.u(Xk, Vi)
        #         numerator += Xk * (u ** self.m)
        #         denominator += u ** self.m
        #     self.centroids[Vi] = numerator / denominator
        self.centroids = np.array([
            np.add.reduce([
                (self.u[d, c] ** self.m) * self.data[d]
                for d in range(len(self.data))])
            / np.sum([
                sample[c] ** self.m
                for sample in self.u])
            for c in range(len(self.centroids))])

    def calc_u(self):
        """
        a method to determine how much data Xk is belonged to cluster Vi

        :param Xk: data
        :param Vi: target cluster
        :return: u
        """
        # Xk_Vi = np.linalg.norm(Xk - Vi)
        # s = 0
        # for Vj in self.centroids:
        #     s += (Xk_Vi / np.linalg.norm(Xk - Vj))**(2 / (self.m - 1))
        # return 1 / s
        self.u = np.array([[
                1 / np.sum(
                    [(np.linalg.norm(Xk - Vi) / np.linalg.norm(Xk - Vj)) ** (2 / (self.m - 1))
                     for Vj in self.centroids]
                )
                for Vi in self.centroids]
            for Xk in self.data], dtype='float64')

    def plotting_clusters(self):
        """
        a method to plot clusters in a scatter diagram (2D)
        :return:
        """
        for cluster, color in zip(self.clusters, self.colors[:self.k]):
            for data_of_cluster in cluster.reshape((-1, 2)):
                plt.scatter(data_of_cluster[0], data_of_cluster[1], c=color, alpha=0.5)
        i = 1
        for centroid, color in zip(self.centroids, self.colors[:self.k]):
            plt.scatter(centroid[0], centroid[1], c=color, edgecolors='w', linewidths=1,
                        s=100, label='centroid #{}\n({}, {})'.format(i, round(centroid[0], 2), round(centroid[1], 2)))
            i += 1

    def calculate_error(self):
        self.error = np.sum([
            np.sum(
                [(self.u[d, c] ** self.m) * (np.linalg.norm(self.data[d] - self.centroids[c]) ** 2)
                 for c in range(len(self.centroids))])
            for d in range(len(self.data))
        ])

    def main(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c='w', alpha=0.5)
        plt.title('RAW DATA BEFORE RUNNING THE ALGORITHM')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


        errors = []
        for i in tqdm(range(self.number_of_iteration)):
            self.calc_u()
            self.number_of_iterations()
            if i != self.number_of_iteration - 1:
                self.update_centroids()
            self.calculate_error()
            errors.append(self.error)
        # plt.plot(errors)
        # plt.show()
        # # self.plotting_clusters()
        # # plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=100, c='w', edgecolors='k')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.legend()

        for j in range(self.k):
            clusterData = np.array([
                self.data[i]
                for i in range(len(self.data))
                if j == np.argmax(self.u[i])
            ])
            print(clusterData)
            plt.scatter(clusterData[:, 0], clusterData[:, 1])
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='black', s=100)
        plt.title('Dataset 1 / m = 10')
        plt.show()

        plt.show()









# print(data)


if __name__ == '__main__':
    FuzzyCMean().main()



