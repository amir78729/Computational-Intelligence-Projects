import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib as npm

def show_image(img):
    image = img[0].reshape((28, 28))
    plt.title('LABEL = {}'.format(np.argmax(img[1])))
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


def get_data(number_of_data, flag):
    # Reading The Train Set
    train_images_file = open('train-images.idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)

    train_labels_file = open('train-labels.idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    if flag:
        num_of_train_images = number_of_samples

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
        num_of_test_images = number_of_samples

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

if __name__ == '__main__':
    print('STEP 1: GETTING THE DATASET')
    number_of_samples = 100
    test_set, train_set = get_data(number_of_samples, True)
    print('\tTRAIN_SET AND TEST_SET ARE READY TO USE!')
    print('\tPLOTTING SOME DATA:\n\t\t',end='')
    number_of_plotting_exampels = 5
    for p in range(number_of_plotting_exampels):
        show_image(train_set[p])
        plt.show()
        print('LABEL={}   '.format(np.argmax(train_set[p][1])), end='')
    print('\n\t\tDONE')

    print('\n', '- ' * 30, '\n')
    print('STEP 2: CALCULATING INITIAL ACCURACY')

    np.set_printoptions(suppress=True)
    learning_rate = 1
    number_of_epochs = 20
    batch_size = 10

    number_of_correct_guesses = 0

    for i in range(number_of_samples):
        """
        + - - - +         + - - - - +         + - - - - +         + - - - - +  
        |  x1   |         |  a1_1   |         |  a2_1   |         |  a3_1   |  
        |  x2   |   w1    |  a1_2   |   w2    |  a2_2   |   w3    |  a3_2   |  
        |  x3   | ----- > |  a1_3   | ----- > |  a2_3   | ----- > |  a3_3   |  
        |  ...  |   b1    |   ...   |   b2    |   ...   |   b3    |   ...   |  
        |  x784 |         |  a1_16  |         |  a2_16  |         |  a3_10  | 
        + - - - +         + - - - - +         + - - - + -         + - - - - + 
        
        """
        w1 = np.random.normal(npm.zeros((16, 784)), npm.ones((16, 784)))
        b1 = np.zeros(16)[np.newaxis]
        x1 = train_set[i][0]
        a1 = calc(w1, x1, b1, "sigmoid")

        w2 = np.random.normal(npm.zeros((16, 16)), npm.ones((16, 16)))
        b2 = np.zeros(16)[np.newaxis]
        a2 = calc(w2, a1, b2, "sigmoid")

        w3 = np.random.normal(npm.zeros((10, 16)), npm.ones((10, 16)))
        b3 = np.zeros(10)[np.newaxis]
        a3 = calc(w3, a2, b3, "sigmoid")

        guess = np.argmax(a3)
        label = np.argmax(train_set[i][1])

        if label == guess:
            number_of_correct_guesses += 1

        # print('{}\t\tGUESS: {}\tLABEL: {}\t{}'.format(i+1, guess, label, label == guess))
    accuracy = number_of_correct_guesses / number_of_samples * 100
    print('\tACCURACY : {}%'.format(accuracy))

