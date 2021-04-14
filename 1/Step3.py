from AdditionalFunctions import *
import numpy.matlib as npm
import time

# initializing wights
w1 = np.random.normal(npm.zeros((16, 784)), npm.ones((16, 784)))
w2 = np.random.normal(npm.zeros((16, 16)), npm.ones((16, 16)))
w3 = np.random.normal(npm.zeros((10, 16)), npm.ones((10, 16)))
# w1 = np.random.randn(16, 784)
# w2 = np.random.randn(16, 16)
# w3 = np.random.randn(10, 16)
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

    def calculate_accuracy(self):
        number_of_correct_guesses = 0
        for image in range(self.number_of_samples):
            guess = np.argmax(self.feedforward(self.train_set[image])[0][-1])
            label = np.argmax(self.train_set[image][1])
            number_of_correct_guesses = number_of_correct_guesses + 1 if guess == label else number_of_correct_guesses
        return number_of_correct_guesses / self.number_of_samples




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
        print('training the network: \n'.upper())
        errors = []
        accuracy = []
        for i in range(self.number_of_epochs):
            epoch_cost = 0
            np.random.shuffle(self.train_set)

            batch_count = 0
            for j in range(0, self.number_of_samples, self.batch_size):
                batch_count += 1
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

                for img in range(self.batch_size):
                    # for img in batch:
                    print('\r\tEPOCH: %02d/%02d\t\tBATCH: %03d/%03d\t\tIMAGE: %04d/%04d\t\t' % (i + 1, self.number_of_epochs, batch_count, number_of_samples // batch_size, img+1, batch_size), end='')
                    a, z = self.feedforward(batch[img])
                    grad_w, grad_b, grad_a = self.back_propagation(grad_w, grad_b, grad_a,  a, z, batch[img])
                    c = 0
                    for x in range(10):
                        c += (batch[img][1][x, 0] - a[2][x, 0]) ** 2
                    epoch_cost += c


                for x in range(3):
                    self.weights[x] -= grad_w[x] / self.batch_size
                    self.biases[x] -= grad_b[x].T / self.batch_size

            errors.append(epoch_cost / number_of_samples)
            accuracy.append(self.calculate_accuracy())
            print('EPOCH COMPLETED!')

        plt.plot(errors, 'g')
        plt.xlabel("Epoch", color='yellow')
        plt.ylabel("Error", color='yellow')
        plt.title('Error in Training Process', color='yellow')
        plt.grid(color='green', linestyle='--', linewidth=0.5)

        plt.show()

        # plt.plot(accuracy, 'r', linestyle="--")
        # plt.xlabel("Epoch", color='orange')
        # plt.ylabel("Error", color='orange')
        # plt.title('Accuracy in Training Process', color='yellow')
        # plt.grid(color='red', linestyle='--', linewidth=0.5)
        # plt.show()

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # # fig.suptitle('Plotting Accuracy and Cost in the Training Process\n')
        # fig.set_size_inches(8.0, 4.0)
        # # fig.tight_layout(pad=10)
        # # fig.subplots_adjust(top=30.0)
        # ax1.plot(errors, 'g')
        # ax1.set(xlabel="Epoch", ylabel='Cost')
        # # ax1.set_title('Error in Training Process', color='yellow')
        # ax1.grid(color='green', linestyle='--', linewidth=0.5)
        #
        # ax2.plot(accuracy, 'r', linestyle="--")
        # ax2.set(xlabel="Epoch", ylabel='Accuracy')
        # # ax2.set_title('Accuracy in Training Process', color='yellow')
        # ax2.grid(color='orange', linestyle='--', linewidth=0.5)
        #
        # # plt.subplots_adjust(top=10)
        #
        # plt.show()

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    plt.style.use("dark_background")
    number_of_samples = 100
    print('step 1: getting the dataset'.upper())
    test_set, train_set = get_data(number_of_samples, True)
    hr()

    learning_rate = 1
    number_of_epochs = 20
    batch_size = 10
    net = NeuralNetwork(learning_rate=learning_rate,
                        number_of_epochs=number_of_epochs,
                        batch_size=batch_size,
                        train_set=train_set,
                        weights=weights,
                        biases=biases,
                        number_of_samples=number_of_samples)

    print('step 2: calculating initial accuracy'.upper())
    print('\tinitial accuracy: {}%'.format(net.calculate_accuracy() * 100))
    hr()
    start_time = time.time()
    net.train_network()
    stop_time = time.time()
    print('\n\ttraining process completed in {}s'.format(round(stop_time - start_time)).upper())
    accuracy = net.calculate_accuracy()
    print('\tthe accuracy of the network is {}%'.format(accuracy * 100).upper())