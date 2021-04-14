import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib as npm


w1 = np.random.normal(npm.zeros((16, 784)), npm.ones((16, 784)))
w2 = np.random.normal(npm.zeros((16, 16)), npm.ones((16, 16)))
w3 = np.random.normal(npm.zeros((10, 16)), npm.ones((10, 16)))
weights = [w1, w2, w3]

# initializing biases
b1 = np.zeros(16)[np.newaxis]
b2 = np.zeros(16)[np.newaxis]
b3 = np.zeros(10)[np.newaxis]
biases = [b1, b2, b3]


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
        z = np.dot(weights, inputs) + biases.T
        return z, sigmoid(z)


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


def calculate_accuracy(guess, label):
    return np.sum(guess == label) / guess.shape[0]


def calculate_cost(a_j, y_j):
    """
    calculating cost using MSE
    :param a_j: guessed values from feedforward
    :param y_j: real values (labels)
    :return: cost = Σ[(a_j - y_j)^2]
    """
    return np.power(np.subtract(a_j, y_j), 2).sum(axis=0)


def feedforward(inputs, activation_function="sigmoid"):
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
    z1, a1 = calc(weights[0], inputs, biases[0], activation_function)
    z2, a2 = calc(weights[1], a1, biases[1], activation_function)
    z3, a3 = calc(weights[2], a2, biases[2], activation_function)
    # print(inputs)
    # print(a3)
    # print('----')
    return [z1, z2, z3], [inputs, a1, a2, a3]


def back_propagate(x, y):
    """
    d(cost)/d(w[L]) = 2 ( a[L] - y ) σ'( z[L] ) a[L-1}
    d(cost)/d(b[L]) = 2 ( a[L] - y ) σ'( z[L] )

    :param x:
    :param y:
    :return:
    """
    db = [np.zeros(b.shape) for b in biases]
    dw = [np.zeros(w.shape) for w in weights]
    z, a = feedforward(x)
    # print(len(a))
    a = a[1:]
    # print(len(a))
    # delta = 2 * (a[-1] - y) * sigmoid_derivative(z[-1])
    delta = 2 * np.subtract(a[-1], y) * sigmoid_derivative(z[-1])
    db[-1] = delta
    dw[-1] = np.dot(delta, a[-2].transpose())

    for layer in range(2, number_of_layers):
        delta = np.dot(weights[-layer + 1].transpose(), delta) * sigmoid_derivative(z[-layer])

        db[-layer] = delta
        dw[-layer] = np.dot(delta, a[-layer - 1].transpose())

        # print(db.shape)
        # print(dw.shape)

    return dw, db, np.argmax(a[2], axis=0)


# def delta_cost_to_delta_b(l, a, y, weights, biases, dc_da):
#     """
#     d(cost)/d(w[L]) = 2 ( a[L] - y ) σ'( z[L] ) a[L-1}
#     :param l:
#     :param a:
#     :param y:
#     :param weights:
#     :param biases:
#     :return:
#     """
#     if layer == number_of_layers:
#         return 2 * np.subtract(a[l], y) * sigmoid_derivative(np.dot(weights[l - 1], a[l - 1]) + biases[l - 1].T)
#     else:
#         dc_da[l] * sigmoid_derivative(np.dot(weights[l - 1], a[l - 1]) + biases[l - 1].T)
#
# def delta_cost_to_delta_w(l, a, y, weights, biases, dc_da):
#     """
#     d(cost)/d(b[L]) = 2 ( a[L] - y ) σ'( z[L] )
#     :param l:
#     :param a:
#     :param y:
#     :param weights:
#     :param biases:
#     :return:
#     """
#     print(np.subtract(a[l], y).shape)
#     print(sigmoid_derivative(np.dot(weights[l - 1], a[l - 1]) + biases[l - 1].T).shape)
#     print(a[l - 1].shape)
#
#     if layer == number_of_layers:
#         return 2 * np.subtract(a[l], y) * sigmoid_derivative(np.dot(weights[l - 1], a[l - 1]) + biases[l - 1].T) * a[l - 1]
#     else:
#         return dc_da[l] * sigmoid_derivative(np.dot(weights[l - 1], a[l - 1]) + biases[l - 1].T) * a[l - 1]
#
# def delta_cost_to_delta_a(l, a, y, weights, biases):
#     """
#     d(cost)/d(b[L]) = 2 ( a[L] - y ) σ'( z[L] )
#     :param l:
#     :param a:
#     :param y:
#     :param weights:
#     :param biases:
#     :return:
#     """
#     return 2 * np.subtract(a[l], y) * sigmoid_derivative(np.dot(weights[l-1], a[l - 1]) + biases[l-1].T) * weights[l-1]

def updateMiniBatch(miniBatch, lr):
    global biases, weights

    nablaB = [np.zeros(b.shape) for b in biases]
    nablaW = [np.zeros(w.shape) for w in weights]

    print(len(nablaB))
    print(len(nablaW))

    cost = 0
    for x, y in miniBatch:
        deltaNablaB, deltaNablaW, output = back_propagate(x, y)

        # print((deltaNablaB.shape))
        # print((deltaNablaW.shape))
        #
        # print((nablaB.shape))
        # print((nablaW.shape))


        # print(len(deltaNablaB))
        # print(len(deltaNablaW))
        #
        # print(len(nablaB))
        # print(len(nablaW))

        cost += (output - y) ** 2
        print('ssssssssssssssssssss')
        # for nb, dnb in zip(nablaB, deltaNablaB):
        #     print(nb.shape)
        #     print(dnb.shape)
        #     print()
        #
        # for nw, dnw in zip(nablaW, deltaNablaW):
        #     print(nw.shape)
        #     print(dnw.shape)
        #     print()

        # nablaB = [nb @ dnb for nb, dnb in zip(nablaB, deltaNablaB)]
        # nablaW = [nw @ dnw for nw, dnw in zip(nablaW, deltaNablaW)]

    weights = [w - lr / len(miniBatch) * nw for w, nw in zip(weights, nablaW)]
    biases = [b - lr / len(miniBatch) * nb for b, nb in zip(biases, nablaB)]

    return cost


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    print('STEP 1: GETTING THE DATASET')
    number_of_samples = 100
    test_set, train_set = get_data(number_of_samples, True)
    test_set, train_set = np.array(test_set), np.array(train_set)

    print('\tTRAIN_SET AND TEST_SET ARE READY TO USE!')
    print('\tPLOTTING SOME DATA:\n\t\t', end='')
    number_of_plotting_examples = 5
    plt.style.use("dark_background")

    # for p in range(number_of_plotting_examples):
    #     show_image(train_set[p])
    #     plt.show()
    #     print('LABEL={}   '.format(np.argmax(train_set[p][1])), end='')
    #
    # print('\n\t\tDONE')

    hr()

    print('STEP 2: CALCULATING FEEDFORWARD OUTPUT AND INITIAL ACCURACY')

    '''
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
    weights = [w1, w2, w3]

    # initializing biases
    b1 = np.zeros(16)[np.newaxis]
    b2 = np.zeros(16)[np.newaxis]
    b3 = np.zeros(10)[np.newaxis]
    biases = [b1, b2, b3]

    number_of_layers = len(weights)

    # calculating output
    a = feedforward(x1, weights, biases, "sigmoid")[3]

    for ooo in a:
        print((ooo.shape))

    # calculating the accuracy
    guess = np.argmax(a, axis=0)
    label = np.argmax(y, axis=0)

    print(guess)
    print(label)

    accuracy = calculate_accuracy(guess, label)
    print('\tACCURACY : {}%'.format(accuracy * 100))
    '''
    label = []
    guess = []

    number_of_layers = len(weights)

    for p in range(100):
        x1 = train_set[p][0]
        y = train_set[p][1]
        # initializing weights

        # calculating output
        a = feedforward(x1, "sigmoid")[1][3]

        # for ooo in a:
        #     print((ooo.shape))

        # calculating the accuracy
        guess.append(np.argmax(a, axis=0))
        label.append(np.argmax(y, axis=0))
        # print("\t\rGuess: %1d - Label: %1d" % (np.argmax(a, axis=0), np.argmax(y, axis=0)), end="")
    guess = np.array(guess)
    label = np.array(label)
    accuracy = calculate_accuracy(guess, label)
    print('\n\tACCURACY : {}%'.format(accuracy * 100))

    hr()

    print('STEP 3: IMPLEMENTING BACK-PROPAGATION')
    # test_set, train_set = np.array(test_set), np.array(train_set)
    # trainingData = list(zip(train_set[:, 0], train_set[:, 1]))

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

    # errors = [0 for x in range(number_of_epochs)]
    errors = []
    for epoch in range(number_of_epochs):
        # shuffle the train set
        np.random.shuffle(train_set)
        cost = 0
        for b in range(0, number_of_samples, batch_size):
            batch = train_set[b:b + batch_size]
            grad_w3 = np.zeros((10, 16))
            grad_w2 = np.zeros((16, 16))
            grad_w1 = np.zeros((16, 784))

            grad_a1 = np.zeros((16, 1))
            grad_a2 = np.zeros((16, 1))
            grad_a3 = np.zeros((10, 1))

            grad_b1 = np.zeros((16, 1))
            grad_b2 = np.zeros((16, 1))
            grad_b3 = np.zeros((10, 1))

            for img in batch:
                # z1 = (w1 @ img[0]) + b1
                # a1 = np.asarray([sigmoid(i) for i in z1]).reshape((16, 1))
                # z2 = (w2 @ a1) + b2
                # a2 = np.asarray([sigmoid(i) for i in z2]).reshape((16, 1))
                # z3 = (w3 @ a2) + b3
                # a3 = np.asarray([sigmoid(i) for i in z3]).reshape((10, 1))

                z1 = (weights[0] @ img[0]) + b1
                a1 = np.asarray([sigmoid(i) for i in z1])
                z2 = (weights[1] @ a1) + b2
                a2 = np.asarray([sigmoid(i) for i in z2])
                z3 = (weights[2] @ a2) + b3
                a3 = np.asarray([sigmoid(i) for i in z3])

                # derivatives
                for x in range(10):
                    for y in range(16):
                        grad_w3[x, y] += a2[y, 0] * sigmoid_derivative(z3[x, 0]) * (2 * a3[x, 0] - 2 * img[1][x, 0])

                for x in range(10):
                    grad_b3[x, 0] += sigmoid_derivative(z3[x, 0]) * (2 * a3[x, 0] - 2 * img[1][x, 0])

                for x in range(16):
                    for y in range(10):
                        grad_a2[x, 0] += w3[y, x] * sigmoid_derivative(z3[y, 0]) * (2 * a3[y, 0] - 2 * img[1][y])

                for x in range(16):
                    for y in range(16):
                        grad_w2[x, y] += grad_a2[x, 0] * sigmoid_derivative(z2[x, 0]) * a1[y, 0]

                for x in range(16):
                    grad_b2[x, 0] += sigmoid_derivative(z2[x, 0]) * grad_a2[x, 0]

                for x in range(16):
                    for y in range(10):
                        grad_a1[x, 0] += w2[y, x] * sigmoid_derivative(z2[y, 0]) * grad_a2[y, 0]

                for x in range(16):
                    for y in range(784):
                        grad_w1[x, y] += grad_a1[x, 0] * sigmoid_derivative(z1[x, 0]) * img[0][y]

                for x in range(16):
                    grad_b1[x, 0] += sigmoid_derivative(z1[x, 0]) * grad_a1[x, 0]

                # cost
                c = 0
                for x in range(10):
                    c += (img[1][x, 0] - a3[x, 0]) ** 2
                cost += c

                # update the w
                # update the b
            weights[2] -= grad_w3 / batch_size
            weights[1] -= grad_w2 / batch_size
            weights[0] -= grad_w1 / batch_size
            biases[2] -= grad_b3 / batch_size
            biases[1] -= grad_b2 / batch_size
            biases[0] -= grad_b1 / batch_size
        print(cost/100)
        errors.append(cost)

    plt.plot(errors)
    plt.show()
        # # shuffle the train set
        # np.random.shuffle(train_set)

        # batches = [trainingData[k: k + batch_size] for k in range(0, len(trainingData), batch_size)]
        #
        # for batch in batches:
        #     costBatch = updateMiniBatch(batch, learning_rate)
        #     errors[epoch] += costBatch
        #
        # errors = [x / len(trainingData) for x in errors]

    # print(errors[1])
    #
    # plt.plot(errors[1])
    # print(errors)
    # plt.show()
        # for batch in range(0, number_of_samples, batch_size):
        #
        #     costBatch = updateMiniBatch(train_set[batch:batch + batch_size], learning_rate)
        #     errors[epoch] += costBatch

            # grad_b = [np.zeros(b.shape) for b in biases]
            # grad_w = [np.zeros(w.shape) for w in weights]
            #
            # mini_batch = train_set[batch:batch + batch_size]
            #
            # label = []
            # guess = []
            #
            # for image in mini_batch:
            #     x = image[0]
            #     y = image[1]
            #
            #     # print(x.shape, y.shape)
            #
            #     z, a = feedforward(x, weights, biases, "sigmoid")
            #     # REMEMBER! len(z)=3 and len(a)=4  (a: [input, a1, a2, a3])
            #
            #     # delta_cost_to_delta_w3 = 2 * np.subtract(a[3], y) * sigmoid_derivative(z[2]) * a[2]
            #     # delta_cost_to_delta_b3 = 2 * np.subtract(a[3], y) * sigmoid_derivative(z[2])
            #
            #     # delta_cost_to_delta_w3 = 2 * np.subtract(a[3]- y) * sigmoid_derivative(z[2]) * a[2]
            #     # delta_cost_to_delta_b3 = 2 * np.subtract(a[3], y) * sigmoid_derivative(z[2])
            #
            #     # for ooo in a:
            #     #     print((ooo.shape))
            #
            #     # calculating the accuracy
            #     guess.append(np.argmax(a[3], axis=0))
            #     label.append(np.argmax(y, axis=0))
            #
            #     # grad_a2 =
            #
            #     # dc_dw = []
            #     # dc_db = []
            #     # dc_da = []
            #     #
            #     # for layer in range(number_of_layers, 0, -1):
            #     #     dc_da.append(delta_cost_to_delta_a(layer, a, y, weights, biases))
            #     #     dc_dw.append(delta_cost_to_delta_w(layer, a, y, weights, biases, dc_da))
            #     #     dc_db.append(delta_cost_to_delta_b(layer, a, y, weights, biases, dc_da))
            #
            #
            #         # print(layer)
            #     # a.insert(0, x)
            #     # guess = np.argmax(a[0])
            #     # label = np.argmax(y)
            #     # print(guess)
            #     # print(label)
            #     # for ooo in a:
            #     #     print((ooo.shape))
            #     # print('-'*20)
            # guess = np.array(guess)
            # label = np.array(label)

            # print(calculate_accuracy(guess, label))
            # print(calculate_cost(guess, label))
            #
            # print('-'*20)

            # print('\tbatch #{}'.format(batch + 1))

            # # creating input and label outputs for first 100 data
            # y = np.zeros(10).reshape(10, 1)
            # x1 = np.zeros(784).reshape(784, 1)
            #
            # for p in range(batch_size):
            #     x1 = np.concatenate((x1, mini_batch[p][0]), axis=1)
            #     y = np.concatenate((y, mini_batch[p][1]), axis=1)
            #
            # # removing the most left column
            # x1 = x1[:, 1:]
            # y = y[:, 1:]
            #
            # # initializing weights
            # # w1 = np.random.normal(npm.zeros((16, 784)), npm.ones((16, 784)))
            # # w2 = np.random.normal(npm.zeros((16, 16)), npm.ones((16, 16)))
            # # w3 = np.random.normal(npm.zeros((10, 16)), npm.ones((10, 16)))
            # #
            # # # initializing biases
            # # b1 = np.zeros(16)[np.newaxis]
            # # b2 = np.zeros(16)[np.newaxis]
            # # b3 = np.zeros(10)[np.newaxis]
            #
            # a = feedforward(x1, weights, biases, "sigmoid")
            # a.insert(0, x1)
            # for ooo in a:
            #     print(len(ooo))
            #
            # guess = np.argmax(a[2], axis=0)
            # label = np.argmax(y, axis=0)
            #
            # cost = calculate_cost(a[3], y)
            #
            # for layer in range(number_of_layers, 0, -1):
            #     dc_dw = delta_cost_to_delta_w(layer, a, y, weights, biases)
            #     dc_dw = delta_cost_to_delta_w(layer, a, y, weights, biases)
            #     # print(layer)
            #
            # # grad_b += 0
            # # grad_w += 0
            # # print(label)
            # # print(guess)
            # # print(a3)
            # # print(y)
            # accuracy = calculate_accuracy(guess, label)
            # print('\t\t', accuracy*100)
            # # print(calculate_cost(a3, y))
            # print('-'*10)
            # a.append(accuracy)

    # plt.plot(a)
    # plt.show()

    hr()






