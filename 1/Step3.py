from AdditionalFunctions import *
from NeuralNetwork import *
import time
from matplotlib.legend_handler import HandlerLine2D


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    plt.style.use("dark_background")
    number_of_samples = 100
    print('step 1: getting the dataset'.upper())
    test_set, train_set = get_data(number_of_samples, True)
    hr()
    activation_function_derivative = activation_functions_derivative['sigmoid']
    activation_function = activation_functions['sigmoid']
    learning_rate = 1
    number_of_epochs = 20
    batch_size = 10
    net = NeuralNetwork(learning_rate=learning_rate,
                        number_of_epochs=number_of_epochs,
                        batch_size=batch_size,
                        train_set=train_set,
                        number_of_samples=number_of_samples,
                        activation_function_derivative=activation_function_derivative,
                        activation_function=activation_function)

    net1 = NeuralNetwork(learning_rate=learning_rate,
                        number_of_epochs=number_of_epochs,
                        batch_size=batch_size,
                        train_set=train_set,
                        number_of_samples=number_of_samples,
                        activation_function_derivative=activation_function_derivative,
                        activation_function=activation_function)

    print('step 2: calculating initial accuracy'.upper())
    print('\tinitial accuracy: {}%'.format(net.calculate_accuracy() * 100))
    hr()
    start_time = time.time()
    net.train_network()
    stop_time = time.time()
    print('\n\ttraining process completed in {}s'.format(round(stop_time - start_time)).upper())
    accuracy = net.calculate_accuracy()
    print('\tthe accuracy of the network is {}%'.format(accuracy * 100).upper())

    # print('step 2: calculating initial accuracy'.upper())
    # print('\tinitial accuracy: {}%'.format(net1.calculate_accuracy() * 100))
    # hr()
    # start_time = time.time()
    # net1.train_network()
    # stop_time = time.time()
    # print('\n\ttraining process completed in {}s'.format(round(stop_time - start_time)).upper())
    # accuracy = net1.calculate_accuracy()
    # print('\tthe accuracy of the network is {}%'.format(accuracy * 100).upper())

    # p1, = plt.plot(e, 'g', label="sigmoid")
    # plt.xlabel("Epoch", color='white')
    # plt.ylabel("Error", color='white')
    # plt.title('Error in Training Process', color='yellow')
    # plt.grid(color='gray', linestyle='--', linewidth=0.5)
    #
    # p2, = plt.plot(e1, 'b', label="tanh")
    # plt.xlabel("Epoch", color='white')
    # plt.ylabel("Error", color='white')
    # plt.title('Error in Training Process', color='yellow')
    # plt.grid(color='gray', linestyle='--', linewidth=0.5)
    #
    # plt.legend(handler_map={p1: HandlerLine2D(numpoints=4)})
    #
    # plt.show()
