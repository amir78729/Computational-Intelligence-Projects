import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt


class NeuralNetworkVectorized:
    def __init__(self, learning_rate, number_of_epochs, batch_size,
                 train_set, number_of_samples,
                 activation_function, activation_function_derivative):
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.set = train_set

        # initializing wights
        w1 = np.random.randn(16, 784)
        w2 = np.random.randn(16, 16)
        w3 = np.random.randn(10, 16)
        w = [w1, w2, w3]

        # initializing biases
        b1 = np.zeros((16, 1))
        b2 = np.zeros((16, 1))
        b3 = np.zeros((10, 1))
        b = [b1, b2, b3]

        self.weights = w
        self.biases = b
        self.number_of_samples = number_of_samples
        self.activation_function = activation_function
        self.activation_function_derivative = np.vectorize(activation_function_derivative)

    def set_set(self, s):
        self.set = s
        self.number_of_samples = len(s)

    def set_number_of_samples(self, s):
        self.number_of_samples = s

    def feedforward(self, img):
        z1 = (self.weights[0] @ img[0]) + self.biases[0]
        a1 = np.asarray([self.activation_function(z[0]) for z in z1]).reshape((16, 1))
        z2 = (self.weights[1] @ a1) + self.biases[1]
        a2 = np.asarray([self.activation_function(z[0]) for z in z2]).reshape((16, 1))
        z3 = (self.weights[2] @ a2) + self.biases[2]
        a3 = np.asarray([self.activation_function(z[0]) for z in z3]).reshape((10, 1))
        return [a1, a2, a3], [z1, z2, z3]

    # def feedforward_vectorized(self, img):
    #     z1 = np.add(np.dot(self.weights[0], img.T), self.biases[0].T)
    #     a1 = self.activation_function(z1)
    #     z2 = np.add(np.dot(self.weights[1], a1), self.biases[1].T)
    #     a2 = self.activation_function(z2)
    #     z3 = np.add(np.dot(self.weights[2], a2), self.biases[2].T)
    #     a3 = self.activation_function(z3)



        # return [np.array(a1), np.array(a2), np.array(a3)], [np.array(z1), np.array(z2), np.array(z3)]

    def back_propagation(self, grad_w, grad_b, grad_a,  a, z, img):
        # print((self.activation_function_derivative(z[2]) * (2 * a[2] - 2 * y)).shape)
        grad_w[2] += (self.activation_function_derivative(z[2]) * (2 * a[2] - 2 * img[1])) @ (np.transpose(a[1]))
        grad_b[2] += (self.activation_function_derivative(z[2]) * (2 * a[2] - 2 * img[1]))

        grad_a[1] += np.transpose(self.weights[2]) @ (self.activation_function_derivative(z[2]) * (2 * a[2] - 2 * img[1]))
        # grad_w[1] += (self.activation_function_derivative(z[1]) * grad_a[1]) @ (np.transpose(a[1]))
        grad_w[1] += (self.activation_function_derivative(z[1]) * a[0]) @ (np.transpose(a[1]))
        grad_b[1] += (self.activation_function_derivative(z[1]) * grad_a[1])

        grad_a[0] += np.transpose(self.weights[1]) @ (self.activation_function_derivative(z[1]) * grad_a[1])
        grad_w[0] += (self.activation_function_derivative(z[0]) * grad_a[0]) @ np.transpose(img[0])
        grad_b[0] += (self.activation_function_derivative(z[0]) * grad_a[0])

        return grad_w, grad_b, grad_a

    def calculate_accuracy(self):
        number_of_correct_guesses = 0
        for image in range(self.number_of_samples):
            guess = np.argmax(self.feedforward(self.set[image])[0][-1])
            label = np.argmax(self.set[image][1])
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
            np.random.shuffle(self.set)

            batch_count = 0
            for j in range(0, self.number_of_samples, self.batch_size):
                batch_count += 1
                batch = self.set[j:j + self.batch_size]

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

                img_index = 0
                for img in batch:
                    # for img in batch:
                    print('\r\tEPOCH: %02d/%02d\t\tBATCH: %03d/%03d\t\tIMAGE: %04d/%04d\t\t' % (i + 1, self.number_of_epochs, batch_count, self.number_of_samples // self.batch_size, img_index+1, self.batch_size), end='')
                    a, z = self.feedforward(img)
                    grad_w, grad_b, grad_a = self.back_propagation(grad_w, grad_b, grad_a,  a, z, img)

                    c = 0
                    for x in range(10):
                        c += (img[1][x, 0] - a[2][x, 0]) ** 2
                    epoch_cost += c

                    # x = img[1] - a[2]
                    # c = np.dot(x.T, x)
                    # epoch_cost += c[0][0]

                    img_index += 1

                for x in range(3):
                    self.weights[x] -= (grad_w[x] / self.batch_size) * self.learning_rate
                    self.biases[x] -= (grad_b[x] / self.batch_size) * self.learning_rate

            errors.append(epoch_cost / self.number_of_samples)

            accuracy.append(self.calculate_accuracy())
            print('EPOCH COMPLETED!')
        return errors

        plt.plot(errors, 'g')
        plt.xlabel("Epoch", color='white')
        plt.ylabel("Error", color='white')
        plt.title('Error in Training Process (Using Vectorization)\n'
                  'Learning rate: {} - Batch Size:{} - '
                  '#Epochs: {}'.format(self.learning_rate, self.batch_size, self.number_of_epochs), color='white')
        plt.grid(color='gray', linestyle='--', linewidth=0.5)

        plt.show()

        return errors

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

    # def train_network_with_vectorization(self):
    #     """
    #         Pseudo Code:
    #             Allocate W matrix and vector b for each layer.
    #             Initialize W from standard normal distribution, and b = 0, for each layer.
    #             Set learning_rate, number_of_epochs, and batch_size.
    #             for i from 0 to number_of_epochs:
    #                 Shuffle the train set.
    #                 for each batch in train set:
    #                     Allocate grad_W matrix and vector grad_b for each layer and initialize to 0.
    #                     for each image in batch:
    #                         Compute the output for this image.
    #                         grad_W += dcost/dW for each layer (using backpropagation)
    #                         grad_b += dcost/db for each layer (using backpropagation)
    #                     W = W - (learning_rate × (grad_W / batch_size))
    #                     b = b - (learning_rate × (grad_b / batch_size))
    #     """
    #     print('training the network: \n'.upper())
    #     errors = []
    #     accuracy = []
    #     for i in range(self.number_of_epochs):
    #         epoch_cost = 0
    #         np.random.shuffle(self.set)
    #
    #         batch_count = 0
    #         for j in range(0, self.number_of_samples, self.batch_size):
    #             batch_count += 1
    #             batch = self.set[j:j + self.batch_size]
    #             # batch = np.array(batch)[:, 0]
    #             # print(batch)
    #
    #             grad_w3 = np.zeros((10, 16))
    #             grad_w2 = np.zeros((16, 16))
    #             grad_w1 = np.zeros((16, 784))
    #             grad_w = [grad_w1, grad_w2, grad_w3]
    #
    #             grad_b1 = np.zeros((16, 1))
    #             grad_b2 = np.zeros((16, 1))
    #             grad_b3 = np.zeros((10, 1))
    #             grad_b = [grad_b1, grad_b2, grad_b3]
    #
    #             grad_a1 = np.zeros((16, 1))
    #             grad_a2 = np.zeros((16, 1))
    #             grad_a3 = np.zeros((10, 1))
    #             grad_a = [grad_a1, grad_a2, grad_a3]
    #
    #             x, y = [], []
    #
    #             for img in range(self.batch_size):
    #                 x.append((batch[img][0]))
    #                 y.append((batch[img][1]))
    #             x, y = np.array(x).reshape(self.batch_size, 784), np.array(y).reshape(self.batch_size, 10)
    #
    #             print(x.shape, y.shape)
    #
    #             # a, z = self.feedforward_vectorized(x)
    #             # print(a[-1])
    #             # print(z.shape)
    #             for img in range(self.batch_size):
    #                 # print(batch.shape)
    #                 print('\r\tEPOCH: %02d/%02d\t\tBATCH: %03d/%03d\t\tIMAGE: %04d/%04d\t\t' % (i + 1, self.number_of_epochs, batch_count, self.number_of_samples // self.batch_size, img+1, self.batch_size), end='')
    #                 a, z = self.feedforward_vectorized(x)
    #                 grad_w, grad_b, grad_a = self.back_propagation_vectorized(grad_w, grad_b, grad_a,  a, z, y)
    #                 c = 0
    #                 for x in range(10):
    #                     c += (batch[img][1][x, 0] - a[2][x, 0]) ** 2
    #                 epoch_cost += c
    #
    #             for x in range(3):
    #                 self.weights[x] -= grad_w[x] / self.batch_size
    #                 self.biases[x] -= grad_b[x].T / self.batch_size
    #
    #         errors.append(epoch_cost / self.number_of_samples)
    #         accuracy.append(self.calculate_accuracy())
    #         print('EPOCH COMPLETED!')
    #
    #     return errors
    #
    #     plt.plot(errors, 'g')
    #     plt.xlabel("Epoch", color='white')
    #     plt.ylabel("Error", color='white')
    #     plt.title('Error in Training Process', color='white')
    #     plt.grid(color='gray', linestyle='--', linewidth=0.5)
    #
    #     plt.show()

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