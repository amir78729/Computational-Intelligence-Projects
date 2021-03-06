import numpy as np
import copy
import matplotlib.pyplot as plt
import math


def show_image(img):
    image = img[0].reshape((28, 28))
    plt.title('LABEL = {}'.format(np.argmax(img[1])))
    plt.imshow(image, 'gray')


def shifted_test_set(test_set):
    res = copy.deepcopy(test_set)
    print('shifting pixels 4 units to the right'.upper())
    for i in range(len(res)):
        print('\r\t%d / %d' % (i + 1, len(res)), end='')
        l = list(res[i])
        l[0] = l[0].reshape((28, 28))
        # plt.imshow(l[0], 'gray')
        # plt.show()
        l[0] = np.roll(l[0], 1, axis=1)
        for j in range(4):
            l[0][:, j] = np.zeros((28,))
        # plt.imshow(l[0], 'gray')
        # plt.show()
        l[0] = np.matrix.flatten(l[0])
        res[i] = tuple(l)
    print('\tDONE')
    return res


def sigmoid(z):
    """
    z is a numpy.ndarray. Returns the sigmoid function for each input element.
    :param z: input
    :return σ(z) = 1 / ( 1 + e^(-1))
    """
    # return np.divide(1.0, np.add(1.0, np.exp(-z)))
    try:
        res = 1 / (1 + math.exp(-z))
    except OverflowError:
        res = 0
    return res


def sigmoid_derivative(z):
    """
    :param z:
    :return: σ'(z) = σ(z)(1 - σ(z))
    """
    s = sigmoid(z)
    return s * (1 - s)
    # return np.multiply(s, np.subtract(1.0, s))


def tanh(z):
    """
    z is a numpy.ndarray. Returns the hyperbolic tangent function for each input element.
    """
    e_pos = np.exp(z)
    e_neg = np.exp(-z)
    return np.divide(np.subtract(e_pos, e_neg), np.add(e_pos, e_neg))


def tanh_derivative(z):
    """
    z is a numpy.ndarray. Returns the derivative of the hyperbolic tangent function for each
    input element.
    """
    f_z = tanh(z)
    return np.subtract(1.0, np.multiply(f_z, f_z))


def relu(z):
    # return np.maximum(0, z)
    if z >= 0:
        return z
    else:
        return 0


def relu_derivative(z):
    # z[z > 0] = 1
    # z[z <= 0] = 0
    # return z
    if z >= 0:
        return 1
    else:
        return 0


activation_functions_derivative = {
    'sigmoid': sigmoid_derivative,
    'tanh': tanh_derivative,
    'relu': relu_derivative
}

activation_functions = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu
}


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
        print('\r\tTRAIN SET: %05d/%05d' % (n+1, num_of_train_images), end='')
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256
        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1
        train_set.append((image, label))
    print('\tDONE')
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
        print('\r\tTEST  SET: %05d/%05d' % (n + 1, num_of_test_images), end='')
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1
        test_set.append((image, label))
    print("\tDONE")
    plt.style.use("dark_background")

    fig, axes = plt.subplots(1, 5)
    fig.set_size_inches(10, 2)

    for ax in range(5):
        image = train_set[ax][0].reshape((28, 28))
        axes[ax].set_title('LABEL = {}'.format(np.argmax(train_set[ax][1])))
        axes[ax].imshow(image, 'gray')
    plt.show()
    return test_set, train_set


def hr():
    print()
    print('- ' * 40, '\n')
