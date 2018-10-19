from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10


# input image dimensions
img_rows, img_cols = 28, 28

def plot_graph(l1, l2, name):
    plt.figure()
    plt.plot(l1, l2, 'g', marker='o', linestyle=':')
    plt.xlabel(name)
    plt.ylabel('Accuracy')
    plt.savefig(name + '.png')

def data(training_size=1.0):

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    num_train_samples_train = x_train.shape[0]
    size = int(training_size * num_train_samples_train)
    x_train = x_train[:size]
    y_train = y_train[:size]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    return x_train, y_train, x_test, y_test, input_shape


# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')



def model(x_train, y_train, x_test, y_test, input_shape, epochs=2, batch_size=128):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    K.clear_session()

    return score[0], score[1]
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

def experiment_training_size():
    sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    accuracy = list()

    for size in sizes:
        x_train, y_train, x_test, y_test, input_shape = data(size)
        test_loss, test_accuracy = model(x_train, y_train, x_test, y_test, input_shape, epochs=10, batch_size=128)
        accuracy.append(test_accuracy)

    plot_graph(sizes, accuracy, 'Varying Training Size')

def experiment_epochs():
    x_train, y_train, x_test, y_test, input_shape = data(training_size=1.0)
    accuracy = list()
    epochs = [1, 5, 7, 9, 10, 12]
    for epoch in epochs:
        test_loss, test_accuracy = model(x_train, y_train, x_test, y_test, input_shape, epochs=epoch, batch_size=128)
        accuracy.append(test_accuracy)

    plot_graph(epochs, accuracy, 'Varying Epochs')

def experiment_batch_size():
    x_train, y_train, x_test, y_test, input_shape = data(training_size=1.0)
    batch_sizes = [10, 20, 50, 70, 100, 120, ]
    accuracy = list()

    for batch_size in batch_sizes:
        test_loss, test_accuracy = model(x_train, y_train, x_test, y_test, input_shape, epochs=10, batch_size=batch_size)
        accuracy.append(test_accuracy)

    plot_graph(batch_sizes, accuracy, 'Varying Batch Size')

def main():
    experiment_training_size()
    experiment_epochs()
    experiment_batch_size()

if __name__ == '__main__':
    main()