import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib as npm

# Activations

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def tanh(x):
  return np.tanh(x)

def relu(x):
  return x if x > 0 else 0

activations = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu
}

# Activation Primes

def sigmoidPrime(x):
  return sigmoid(x) * (1 - sigmoid(x))

def tanhPrime(x):
  return 1 / np.cosh(-x)

def reluPrime(x):
  return 1 if x > 0 else 0

primes = {
    'sigmoid': sigmoidPrime,
    'tanh': tanhPrime,
    'relu': reluPrime
}



# Activation
activationFunction = 'sigmoid'
activationVector = np.vectorize(activations[activationFunction])
primeVector = np.vectorize(primes[activationFunction])

# Size
size = [784, 16, 16, 10]
layers = len(size)

# Bias and Weight
biases = [np.zeros((x, 1)) for x in size[1: ]]
weights = [np.random.standard_normal(size=(y, x)) for x, y in zip(size[: -1], size[1: ])]

def show_image(img):
    image = img[0].reshape((28, 28))
    plt.title('LABEL = {}'.format(np.argmax(img[1])))

    # plt.gcf().set_
    plt.imshow(image, 'gray')


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



def backpropagation(x, y):
  db = [np.zeros(b.shape) for b in biases]
  dw = [np.zeros(w.shape) for w in weights]

  z_s, a_s, output = feedforward(x.reshape(784, 1))

  delta = (a_s[-1] - y) * primeVector(z_s[-1])
  db[-1] = delta
  dw[-1] = np.dot(delta, a_s[-2].transpose())

  for l in range(2, layers):
    delta = np.dot(weights[-l + 1].transpose(), delta) * primeVector(z_s[-l])

    db[-l] = delta
    dw[-l] = np.dot(delta, a_s[-l - 1].transpose())

  return db, dw, output


def updateMiniBatch(miniBatch, lr):
    global biases, weights
    nablaB = [np.zeros(b.shape) for b in biases]
    nablaW = [np.zeros(w.shape) for w in weights]

    cost = 0
    for x, y in miniBatch:
        deltaNablaB, deltaNablaW, output = backpropagation(x, y)
        cost += (output - y) ** 2

        nablaB = [nb + dnb for nb, dnb in zip(nablaB, deltaNablaB)]
        nablaW = [nw + dnw for nw, dnw in zip(nablaW, deltaNablaW)]

    weights = [w - lr / len(miniBatch) * nw for w, nw in zip(weights, nablaW)]
    biases = [b - lr / len(miniBatch) * nb for b, nb in zip(biases, nablaB)]

    return cost


def train(X, y, batchSize=20, epochs=100, lr=0.1):
    trainingData = list(zip(X, y))

    errors = [0 for x in range(epochs)]
    for j in range(epochs):
        np.random.shuffle(trainingData)

        batches = [trainingData[k: k + batchSize] for k in range(0, len(trainingData), batchSize)]

        for batch in batches:
            costBatch = updateMiniBatch(batch, lr)
            errors[j] += costBatch

    errors = [x / len(trainingData) for x in errors]

    plt.plot(errors)
    print(errors)
    plt.show()


def feedforward(x):
    activation = x
    ar = [x]
    zs = []

    for b, w in zip(biases, weights):
        z = np.dot(w, activation) + b
        zs.append(z)

        activation = activationVector(z)
        ar.append(activation)
        return zs, ar, np.argmax(ar[-1])

def getFinal(x):
    _, _, output = feedforward(x)
    return output


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    print('STEP 1: GETTING THE DATASET')
    number_of_samples = 100

    test_set, train_set = get_data(number_of_samples, True)
    print('\tTRAIN_SET AND TEST_SET ARE READY TO USE!')
    print('\tPLOTTING SOME DATA:\n\t\t', end='')
    number_of_plotting_examples = 5

    # b = np.zeros(10).reshape(10, 1)

    # print(b)
    plt.style.use("dark_background")
    for p in range(number_of_plotting_examples):
    show_image(train_set[p])
    plt.show()
    print('LABEL={}   '.format(np.argmax(train_set[p][1])), end='')

    print('\n\t\tDONE')