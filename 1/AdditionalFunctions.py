# from read_dataset import *
# import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math



def show_image(img):
    image = img[0].reshape((28, 28))
    plt.title('LABEL = {}'.format(np.argmax(img[1])))
    plt.imshow(image, 'gray')


def sigmoid(z):
    """
    z is a numpy.ndarray. Returns the sigmoid function for each input element.
    :param z: input
    :return σ(z) = 1 / ( 1 + e^(-1))
    """
    # return np.divide(1.0, np.add(1.0, np.exp(-z)))
    return 1 / (1 + math.exp(-z))

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


def relu():
    pass

def relu_derivative():
    pass


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






