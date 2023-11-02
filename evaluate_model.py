import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import sys
import itertools
import argparse


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from keras.datasets import fashion_mnist
from keras.utils import to_categorical

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


labels={0:"T-shirt/top",1:"Trouser",2:"Pullover",3:"Dress",4:"Coat",5:"Sandal",6:"Shirt",7:"Sneaker",8:"Bag",9:"Ankle boot"}
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_data(folder_path):

    global X_train, y_train, X_test, y_test
    train_data = pd.read_csv(os.path.join(
        folder_path, 'mnist/fashion-mnist_train.csv'))
    test_data = pd.read_csv(os.path.join(
        folder_path, 'mnist/fashion-mnist_test.csv'))

    # Extract features (pixels) and labels from the data
    X_train = train_data.iloc[:, 1:].values.astype('float32') / 255  # Normalize pixel values to 0-1
    y_train = train_data.iloc[:, 0].values

    X_test = test_data.iloc[:, 1:].values.astype('float32') / 255  # Normalize pixel values to 0-1
    y_test = test_data.iloc[:, 0].values

    # One-hot encode the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Reshape features to 4D arrays (batch_size, height, width, channels)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    return X_train, y_train, X_test, y_test


def data_analysis():
    print('Training data shape : ', X_train.shape, y_train.shape)
    print('Testing data shape : ', X_test.shape, y_test.shape)
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.figure(figsize=(7, 7))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(X_train[i], cmap='gist_yarg')
        class_index = np.argmax(y_train[i])  # Get the index of the class (position of 1 in one-hot vector)
        class_label = class_names[class_index]  # Map index to class name
        plt.title(class_label)
        plt.axis('off')
   
    plt.savefig("plots/Train_images.png")
    # # how many images the training dataset includes for each label.
    # unique_values, counts = np.unique(y_train, return_counts=True)
    # # Print unique values and their counts
    # for value, count in zip(unique_values, counts):
    #     print(f"Value: {value}, Count: {count}")
    # # check null values
    # print(sum(np.isnan(y_train)))
    # print(sum(np.isnan(y_test)))
    


def define_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))  # input layer
    model.add(Dense(512, activation='relu'))  # hidden layer
    model.add(Dense(10, activation='softmax'))  # output layer
    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def evaluate_model(model, trainX, trainY, testX, testY):
    
    scores, histories = list(), list()

    # fit model
    history = model.fit(trainX, trainY, epochs=5, batch_size=32,
                        validation_data=(testX, testY), verbose=0)
    model.save('final_model.keras')
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)

    print('> %.3f' % (acc * 100.0))
    # append scores
    scores.append(acc)
    histories.append(history)
    return scores, histories
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    plt.savefig("confusion_matrix.png")

def save_summary(model, metric, output_path):
    with open(os.path.join(output_path, 'output.txt'), 'w') as f:
        f.write("Model Architecture:\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f"\nEvaluation Accuracy: {metric:.2f}\n")
        # Add additional insights or observations if necessary


def main(data_folder):
    try:
        # Load and preprocess data
        X_train, y_train, X_test, y_test = load_data(data_folder)

        data_analysis()

        model = load_model('final_model.keras')
        scores, histories = evaluate_model(model, X_train, y_train, X_test, y_test)
        print(scores)
        #print(histories)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

        # Load pre-trained model
        # model = load_model('your_model_path.h5')  # Provide the path to your pre-trained model

        # Evaluate the model
        # accuracy = evaluate_model(model, X_test, y_test)

        # Save model summary and evaluation metric to output.txt
        # save_summary(model, accuracy, data_folder)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py <path_to_data_folder>")
        sys.exit(1)

    data_folder = sys.argv[1]

    if not os.path.exists(data_folder) or not os.listdir(data_folder):
        print("Error: Provided folder doesn't exist or is empty.")
        sys.exit(1)

    main(data_folder)
