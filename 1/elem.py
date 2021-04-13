import math
from read_dataset import *
import random


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def guess_the_number(mat):
    max = -10000
    index = 0
    for i in range(len(mat)):
        if mat[i] > max:
            max = mat[i]
            index = i
    return index


# create the matrix of weights randomly and the biases.
w1 = np.random.randn(16, 784)
w2 = np.random.randn(16, 16)
w3 = np.random.randn(10, 16)
b1 = np.zeros((16, 1))
b2 = np.zeros((16, 1))
b3 = np.zeros((10, 1))

learning_rate = 1
epoch_num = 20
batch_size = 10
right_guesses = 0
train = train_set[:100]
costs = []
print(len(train))

for i in range(epoch_num):
    cost = 0
    random.shuffle(train)
    j = 0
    while j < 100:
        # print(j, j+10)
        batch = train[j: j+10]
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
            z1 = (w1 @ img[0]) + b1
            a1 = np.asarray([sigmoid(i) for i in z1]).reshape((16, 1))
            z2 = (w2 @ a1) + b2
            a2 = np.asarray([sigmoid(i) for i in z2]).reshape((16, 1))
            z3 = (w3 @ a2) + b3
            a3 = np.asarray([sigmoid(i) for i in z3]).reshape((10, 1))

            # derivatives
            for x in range(10):
                for y in range(16):
                    grad_w3[x, y] += a2[y, 0] * sigmoid_derivative(z3[x, 0]) * (2 * a3[x, 0] - 2 * img[1][x, 0])

            for x in range(10):
                grad_b3[x, 0] += sigmoid_derivative(z3[x, 0]) * (2 * a3[x, 0] - 2 * img[1][x, 0])

            for x in range(16):
                for y in range(10):
                    grad_a2[x, 0] += w3[y, x] * sigmoid_derivative(z3[y, 0]) * (2 * a3[y, 0] - 2 * img[1][y, 0])

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
                    grad_w1[x, y] += grad_a1[x, 0] * sigmoid_derivative(z1[x, 0]) * img[0][y, 0]

            for x in range(16):
                grad_b1[x, 0] += sigmoid_derivative(z1[x, 0]) * grad_a1[x, 0]

            # cost
            c = 0
            for x in range(10):
                c += (img[1][x, 0] - a3[x, 0]) ** 2
            cost += c
        j += 10
        # update the w
        # update the b
        w3 -= grad_w3 / 100
        w2 -= grad_w2 / 100
        w1 -= grad_w1 / 100
        b3 -= grad_b3 / 100
        b2 -= grad_b2 / 100
        b1 -= grad_b1 / 100

    print("average cost:", cost / 100)
    costs.append(cost)


for i in range(100):
    z1 = (w1 @ train_set[i][0]) + b1
    a1 = np.asarray([sigmoid(i) for i in z1]).reshape((16, 1))
    z2 = (w2 @ a1) + b2
    a2 = np.asarray([sigmoid(i) for i in z2]).reshape((16, 1))
    z3 = (w3 @ a2) + b3
    a3 = np.asarray([sigmoid(i) for i in z3]).reshape((10, 1))
    print(guess_the_number(a3), guess_the_number(train_set[i][1]))
    if guess_the_number(a3) == guess_the_number(train_set[i][1]):
        right_guesses += 1

print("the accuracy of this model is:", right_guesses / 100)
x = [i+1 for i in range(epoch_num)]
plt.plot(x, costs)
plt.show()