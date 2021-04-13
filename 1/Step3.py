from AdditionalFunctions import *
import numpy.matlib as npm
import numpy
import random

# initializing wights
# w1 = np.random.normal(npm.zeros((16, 784)), npm.ones((16, 784)))
# w2 = np.random.normal(npm.zeros((16, 16)), npm.ones((16, 16)))
# w3 = np.random.normal(npm.zeros((10, 16)), npm.ones((10, 16)))
w1 = np.random.randn(16, 784)
w2 = np.random.randn(16, 16)
w3 = np.random.randn(10, 16)
weights = [w1, w2, w3]

# initializing biases
b1 = np.zeros(16)[np.newaxis]
b2 = np.zeros(16)[np.newaxis]
b3 = np.zeros(10)[np.newaxis]
biases = [b1, b2, b3]


class NeuralNetwork:
    def __init__(self, learning_rate, number_of_epochs, batch_size, train_set, weights, biases, number_of_samples):
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.train_set = train_set
        self.weights = weights
        self.biases = biases
        self.number_of_samples = number_of_samples

    def feedforward(self, img):
        z1 = (weights[0] @ img[0]) + biases[0]

        # for z in z1:
        #     print(z)

        a1 = np.asarray([sigmoid(z[0]) for z in z1]).reshape((16, 1))
        z2 = (weights[1] @ a1) + biases[1]
        a2 = np.asarray([sigmoid(z[0]) for z in z2]).reshape((16, 1))
        z3 = (weights[2] @ a2) + biases[2]
        a3 = np.asarray([sigmoid(z[0]) for z in z3]).reshape((10, 1))
        return [a1, a2, a3], [z1, z2, z3]

    def back_propagation(self, grad_w, grad_b, grad_a,  a, z, img):
        for x in range(10):
            for y in range(16):
                grad_w[2][x, y] += a[1][y, 0] * sigmoid_derivative(z[2][x, 0]) * (2 * a[2][x, 0] - 2 * img[1][x, 0])

        for x in range(10):
            grad_b[2][x, 0] += sigmoid_derivative(z[2][x, 0]) * (2 * a[2][x, 0] - 2 * img[1][x, 0])

        for x in range(16):
            for y in range(10):
                grad_a[1][x, 0] += w3[y, x] * sigmoid_derivative(z[2][y, 0]) * (2 * a[2][y, 0] - 2 * img[1][y])

        for x in range(16):
            for y in range(16):
                grad_w[1][x, y] += grad_a[1][x, 0] * sigmoid_derivative(z[1][x, 0]) * a[0][y, 0]

        for x in range(16):
            grad_b[1][x, 0] += sigmoid_derivative(z[1][x, 0]) * grad_a[1][x, 0]

        for x in range(16):
            for y in range(10):
                grad_a[0][x, 0] += w2[y, x] * sigmoid_derivative(z[1][y, 0]) * grad_a[1][y, 0]

        for x in range(16):
            for y in range(784):
                grad_w[0][x, y] += grad_a[0][x, 0] * sigmoid_derivative(z[0][x, 0]) * img[0][y]

        for x in range(16):
            grad_b[0][x, 0] += sigmoid_derivative(z[0][x, 0]) * grad_a[0][x, 0]

        return grad_w, grad_b, grad_a

    def update(self):
        pass

    def train_network(self):
        """
            Pseudo Code:
                Allocate W matrix and vector b for each layer.
                Initialize W from standard normal distribution, and b = 0, for each layer.
                Set learning_rate, number_of_epochs, and batch_size.
                for i from 0 to number_of_epochs:
                    Shuffle the train set.
                    for each batch in train set:
                        Allocate grad_W matrix and vector grad_b for each layer and initialize to 0.
                        for each image in batch:
                            Compute the output for this image.
                            grad_W += dcost/dW for each layer (using backpropagation)
                            grad_b += dcost/db for each layer (using backpropagation)
                        W = W - (learning_rate × (grad_W / batch_size))
                        b = b - (learning_rate × (grad_b / batch_size))
        """
        errors = []
        for i in range(self.number_of_epochs):
            print('\repoch %d' % (i+1), end='')
            epoch_cost = 0
            np.random.shuffle(self.train_set)
            for j in range(0, self.number_of_samples, self.batch_size):
                batch = self.train_set[j:j + self.batch_size]

                grad_w3 = np.zeros((10, 16))
                grad_w2 = np.zeros((16, 16))
                grad_w1 = np.zeros((16, 784))
                grad_w = [grad_w1, grad_w2, grad_w3]

                grad_b1 = np.zeros((16, 1))
                grad_b2 = np.zeros((16, 1))
                grad_b3 = np.zeros((10, 1))
                grad_b = [grad_b1, grad_b2, grad_b3]

                grad_a1 = np.zeros((16, 1))
                grad_a2 = np.zeros((16, 1))
                grad_a3 = np.zeros((10, 1))
                grad_a = [grad_a1, grad_a2, grad_a3]

                for img in batch:
                    a, z = self.feedforward(img)
                    grad_w, grad_b, grad_a = self.back_propagation(grad_w, grad_b, grad_a,  a, z, img)
                    c = 0
                    for x in range(10):
                        c += (img[1][x, 0] - a[2][x, 0]) ** 2
                    epoch_cost += c

                # self.weights[2] -= grad_w[2] / self.batch_size
                # self.weights[1] -= grad_w[1] / self.batch_size
                # self.weights[0] -= grad_w[0] / self.batch_size
                #
                # self.biases[2] -= grad_b[2].T / self.batch_size
                # self.biases[1] -= grad_b[1].T / self.batch_size
                # self.biases[0] -= grad_b[0].T / self.batch_size

                for x in range(3):
                    self.weights[x] -= grad_w[x] / self.batch_size
                    self.biases[x] -= grad_b[x].T / self.batch_size


            errors.append(epoch_cost)
        plt.plot(errors, 'g')
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.title('Error in Training Process')
        plt.grid(color='green', linestyle='--', linewidth=0.5)
        plt.show()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    number_of_samples = 100
    test_set, train_set = get_data(number_of_samples, True)
    # test_set, train_set = np.array(test_set), np.array(train_set)

    plt.style.use("dark_background")


    learning_rate = 1
    number_of_epochs = 5
    batch_size = 10
    number_of_samples = 5
    net = NeuralNetwork(learning_rate=learning_rate,
                        number_of_epochs=number_of_epochs,
                        batch_size=batch_size,
                        train_set=train_set,
                        weights=weights,
                        biases=biases,
                        number_of_samples=number_of_samples)
    net.train_network()
