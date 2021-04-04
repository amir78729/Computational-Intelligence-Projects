import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib as npm


class NeuralNetwork:
    def __init__(self, sizes, activation_func, activation_func_derivative, initial_weights_func):
        self.sizes = sizes
        self.activation_func = activation_func
        self.activation_func_derivative = activation_func_derivative
        pass


    def feedforward(self, x):
        pass


def show_image(img):
    image = img[0].reshape((28, 28))
    plt.title('LABEL = {}'.format(np.argmax(img[1])))

    # plt.gcf().set_
    plt.imshow(image, 'gray')


def calc(weights, inputs, biases, activation_function):
    """
    calculating output of a layer using INPUTS, WEIGHT matrix and ACTIVATION FUNCTION
    :param weights: a m x n matrix
    :param inputs: a n×1 vector
    :param biases: a m×1 vector
    :param activation_function: usually a sigmoid function
    :return: output = σ( W × A + B)
    """
    if activation_function == "sigmoid":
        return sigmoid(np.dot(weights, inputs) + biases.T)


def sigmoid(z):
    """
    z is a numpy.ndarray. Returns the sigmoid function for each input element.
    :param z: input
    :return σ(z) = 1 / ( 1 + e^(-1))
    """
    return np.divide(1.0, np.add(1.0, np.exp(-z)))


def sigmoid_derivative(z):
    """
    :param z:
    :return: σ'(z) = σ(z)(1 - σ(z))
    """
    s = sigmoid(z)
    return np.multiply(s, np.subtract(1.0, s))


def get_data(number_of_data, flag):
    # Reading The Train Set
    train_images_file = open('train-images.idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)

    train_labels_file = open('train-labels.idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    if flag:
        num_of_train_images = number_of_data

    train_set = []
    for n in range(num_of_train_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1


        train_set.append((image, label))
        # print(train_set[n][1])



    # Reading The Test Set
    test_images_file = open('t10k-images.idx3-ubyte', 'rb')
    test_images_file.seek(4)

    test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
    test_labels_file.seek(8)

    num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
    test_images_file.seek(16)

    if flag:
        num_of_test_images = number_of_data

    test_set = []
    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        test_set.append((image, label))
    return test_set, train_set


def hr():
    print('\n', '- ' * 40, '\n')


def calculate_accuracy(guess, lable):
    return np.sum(guess == label) / guess.shape[0]


def calculate_cost(a_j, y_j):
    """
    calculating cost using MSE
    :param a_j: guessed values from feedforward
    :param y_j: real values (labels)
    :return: cost = Σ[(a_j - y_j)^2]
    """
    return np.power(np.subtract(a_j, y_j), 2).sum(axis=0)


def feedforward(inputs, weight, bias, activation_function="sigmoid"):
    """
    + - - - +         + - - - - +         + - - - - +         + - - - - +
    |  x1   |         |  a1_1   |         |  a2_1   |         |  a3_1   |
    |  x2   |   w1    |  a1_2   |   w2    |  a2_2   |   w3    |  a3_2   |
    |  x3   | ----- > |  a1_3   | ----- > |  a2_3   | ----- > |  a3_3   |
    |  ...  |   b1    |   ...   |   b2    |   ...   |   b3    |   ...   |
    |  x784 |         |  a1_16  |         |  a2_16  |         |  a3_10  |
    + - - - +         + - - - - +         + - - - + -         + - - - - +

    :param inputs:
    :param weight:
    :param bias:
    :param activation_function:
    :return:
    """
    a1 = calc(weight[0], inputs, bias[0], activation_function)
    a2 = calc(weight[1], a1, bias[1], activation_function)
    return calc(weight[2], a2, bias[2], activation_function)


def train_network(net, data, epoch_count, batch_size, eta, error_rate_func=None):

    n = len(data)
    for e in range(epoch_count):
        np.random.shuffle(data)
        for k in range(0, n, batch_size):
            mini_batch = data[k:k + batch_size]
            net.update(mini_batch, eta)
            print("\rEpoch %02d, %05d instances" % (e + 1, k + batch_size), end="")
        print()
        if error_rate_func:
            error_rate = error_rate_func(net)
            print("Epoch %02d, error rate = %.2f" % (e + 1, error_rate * 100))


if __name__ == '__main__':
    print('STEP 1: GETTING THE DATASET')
    number_of_samples = 100

    test_set, train_set = get_data(number_of_samples, True)
    print('\tTRAIN_SET AND TEST_SET ARE READY TO USE!')
    print('\tPLOTTING SOME DATA:\n\t\t',end='')
    number_of_plotting_examples = 5

    b = np.zeros(10).reshape(10, 1)

    # print(b)
    plt.style.use("dark_background")
    for p in range(number_of_plotting_examples):
        show_image(train_set[p])
        plt.show()
        print('LABEL={}   '.format(np.argmax(train_set[p][1])), end='')

        # print(train_set[p][1])
        b = np.concatenate((b, train_set[p][1]), axis=1)

    b = b[:, 1:]  # remove the 1st column with zeros

    print('\n\t\tDONE')

    hr()

    print('STEP 2: CALCULATING FEEDFORWARD OUTPUT AND INITIAL ACCURACY')

    np.set_printoptions(suppress=True)

    # a3 = []
    # cost = []

    # creating input and label outputs for first 100 data
    y = np.zeros(10).reshape(10, 1)
    x1 = np.zeros(784).reshape(784, 1)
    for p in range(number_of_samples):
        x1 = np.concatenate((x1, train_set[p][0]), axis=1)
        y = np.concatenate((y, train_set[p][1]), axis=1)

    # removing the most left column
    x1 = x1[:, 1:]
    y = y[:, 1:]

    # initializing weights
    w1 = np.random.normal(npm.zeros((16, 784)), npm.ones((16, 784)))
    w2 = np.random.normal(npm.zeros((16, 16)), npm.ones((16, 16)))
    w3 = np.random.normal(npm.zeros((10, 16)), npm.ones((10, 16)))

    # initializing biases
    b1 = np.zeros(16)[np.newaxis]
    b2 = np.zeros(16)[np.newaxis]
    b3 = np.zeros(10)[np.newaxis]

    # calculating output
    a3 = feedforward(x1, [w1, w2, w3], [b1, b2, b3], "sigmoid")

    # calculating the accuracy
    guess = np.argmax(a3, axis=0)
    label = np.argmax(y, axis=0)
    accuracy = calculate_accuracy(guess, label)
    print('\tACCURACY : {}%'.format(accuracy * 100))

    hr()

    print('STEP 3: IMPLEMENTING BACK-PROPAGATION')

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

    learning_rate = 1
    number_of_epochs = 20
    batch_size = 10

    for epoch in range(number_of_epochs):
        print('epoch #{}'.format(epoch + 1))

        # shuffle the train set
        np.random.shuffle(train_set)

        for batch in range(number_of_samples//batch_size):
            print('\tbatch #{}'.format(batch + 1))

            # creating input and label outputs for first 100 data
            y = np.zeros(10).reshape(10, 1)
            x1 = np.zeros(784).reshape(784, 1)

            # initializing weights
            w1 = np.random.normal(npm.zeros((16, 784)), npm.ones((16, 784)))
            w2 = np.random.normal(npm.zeros((16, 16)), npm.ones((16, 16)))
            w3 = np.random.normal(npm.zeros((10, 16)), npm.ones((10, 16)))

            # initializing biases
            b1 = np.zeros(16)[np.newaxis]
            b2 = np.zeros(16)[np.newaxis]
            b3 = np.zeros(10)[np.newaxis]

            for p in range(batch_size):
                x1 = np.concatenate((x1, train_set[batch + p][0]), axis=1)
                y = np.concatenate((y, train_set[batch + p][1]), axis=1)

            # removing the most left column
            x1 = x1[:, 1:]
            y = y[:, 1:]

            a3 = feedforward(x1, [w1, w2, w3], [b1, b2, b3], "sigmoid")

            guess = np.argmax(a3, axis=0)
            label = np.argmax(y, axis=0)
            print(label)
            accuracy = calculate_accuracy(guess, label)
            print('\t\t', accuracy*100)
            # print(calculate_cost(a3, y))


    hr()





