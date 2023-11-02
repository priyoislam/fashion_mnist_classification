
import os
import sys
import random


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tabulate import tabulate


from sklearn.metrics import confusion_matrix, classification_report

from keras.datasets import fashion_mnist
from keras.utils import to_categorical

#import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


labels = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
          4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_data(folder_path):

    global X_train, y_train, X_test, y_test, unique_labels, label_counts

    train_data = pd.read_csv(os.path.join(
        folder_path, 'fashion-mnist_train.csv'))
    test_data = pd.read_csv(os.path.join(
        folder_path, 'fashion-mnist_test.csv'))

    # Extract features (pixels) and labels from the data
    X_train = train_data.iloc[:, 1:].values.astype('float32') / 255     # Normalize pixel values to 0-1
    y_train = train_data.iloc[:, 0].values

    unique_labels, label_counts = np.unique(y_train, return_counts=True)

    X_test = test_data.iloc[:, 1:].values.astype('float32') / 255  # Normalize pixel values to 0-1
    y_test = test_data.iloc[:, 0].values

    # One-hot encode the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Reshape features to 4D arrays (batch_size, height, width, channels)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)



def data_analysis():

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.figure(figsize=(7, 7))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(X_train[i], cmap='gist_yarg')
        # Get the index of the class (position of 1 in one-hot vector)
        class_index = np.argmax(y_train[i])
        class_label = class_names[class_index]  # Map index to class name
        plt.title(class_label)
        plt.axis('off')

    plt.savefig("plots/Train_images.png")

    label_list = []

        # Print unique values and their counts and add them to the list
    for label, count in zip(unique_labels, label_counts):
        label_list.append((labels[label], count))

    label_table = tabulate(label_list, headers=['Class', 'Count'], tablefmt='grid')

    return label_table



def define_model():

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))  # input layer
    model.add(Dense(512, activation='relu'))  # hidden layer
    model.add(Dense(10, activation='softmax'))  # output layer
    # model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def evaluate_model(model):

    # fit model
    history = model.fit(X_train, y_train, epochs=5, batch_size=32,validation_split=0.2, verbose=0)
    
    train_loss = history.history['loss'][-1]
    train_accuracy = history.history['accuracy'][-1]
    model.save('final_model.keras')
    
    
    # evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)


    pd.DataFrame(history.history).plot()
    plt.xlabel('epoch')
    plt.savefig("plots/Accuracy_loss.png")
    
    return train_loss, train_accuracy, test_loss, test_accuracy



def save_summary(lable_table, model, cr, mtx,train_loss, train_accuracy, test_loss, test_accuracy):

    with open('output.txt', 'w') as f:
        f.write("Data Analysis:\n\n")
        f.write(f'Training data shape: {X_train.shape}, {y_train.shape}\n')
        f.write(f'Testing data shape: {X_test.shape}, {y_test.shape}\n')
        f.write("\nDistribution of Data in Training Dataset: \n")
        f.write(lable_table)
        f.write("\n")
        f.write("\nFor training image samples check plots/Train_images.png \n\n")
        f.write("\nModel Architecture:\n\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n\n\nTraining Accuracy :")
        f.write(train_accuracy)
        f.write("\nTraining Loss :")
        f.write(train_loss)
        f.write("\nValidation Accuracy :")
        f.write(test_accuracy)
        f.write("\nValidation Loss :")
        f.write(test_loss)
        f.write("\n\nClassifiaction Report:\n")
        f.write(cr)
        f.write("\nConfusion Matrix:\n")
        f.write(mtx)



def main(data_folder):
    try:
        
        load_data(data_folder)

        lable_table = data_analysis()

        model = load_model('final_model.keras')
        train_loss, train_accuracy, test_loss, test_accuracy = evaluate_model(model)

        train_loss_percent = "{:.2f}%".format(train_loss * 100)
        train_accuracy_percent = "{:.2f}%".format(train_accuracy * 100)
        test_loss_percent = "{:.2f}%".format(test_loss * 100)
        test_accuracy_percent = "{:.2f}%".format(test_accuracy * 100)

        
        # Model Prediction
        Y_pred = model.predict(X_test)
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        Y_true = np.argmax(y_test, axis=1)


        # compute the confusion matrix
        confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

        confusion_table = tabulate(confusion_mtx, headers=class_names, showindex=class_names, tablefmt='grid')


        # Classifiaction Report
        cr = classification_report(Y_true, Y_pred_classes, target_names=class_names)

        save_summary(lable_table,model, cr, confusion_table,train_loss_percent, train_accuracy_percent, test_loss_percent, test_accuracy_percent)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py <path_to_data_folder>")
        sys.exit(1)

    data_folder = sys.argv[1]

    if not os.path.exists(data_folder) or not os.listdir(data_folder):
        print("Error: Provided folder doesn't exist or is empty.")
        sys.exit(1)

    main(data_folder)
