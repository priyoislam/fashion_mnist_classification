import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import random
import os
import sys
import itertools
import argparse
from tabulate import tabulate


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

    global X_train, y_train, X_test, y_test,unique_labels, label_counts
    train_data = pd.read_csv(os.path.join(
        folder_path, 'mnist/fashion-mnist_train.csv'))
    test_data = pd.read_csv(os.path.join(
        folder_path, 'mnist/fashion-mnist_test.csv'))

    # Extract features (pixels) and labels from the data
    X_train = train_data.iloc[:, 1:].values.astype('float32') / 255  # Normalize pixel values to 0-1
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

    #return X_train, y_train, X_test, y_test


def data_analysis():

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

    
    # Initialize an empty list to store the tuples
  
    
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
    #model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def evaluate_model(model):

    
    # fit model
    history = model.fit(X_train, y_train, epochs=5, batch_size=32,
                        validation_split=0.2, verbose=0)
    model.save('final_model.keras')
    # evaluate model
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    pd.DataFrame(history.history).plot()
    plt.xlabel('epoch')
    plt.savefig("plots/Accuracy_loss.png")
    #print('> %.3f' % (acc * 100.0))
    final_training_loss = history.history['loss'][-1]
    # scores.append(acc)
    # histories.append(history)
    return acc, final_training_loss 


    plt.grid(False)
    plt.xticks(range(10), class_names, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap='Blues'):
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
    plt.savefig("plots/confusion_matrix.png")

def format_confusion_matrix(confusion_matrix, class_names):
            matrix_str = 'Actual / Predicted\t' + '\t'.join(class_names) + '\n'
            for i, class_name in enumerate(class_names):
                row_str = class_name + '\t\t\t'
                for j in range(len(class_names)):
                    row_str += str(confusion_matrix[i][j]) + '\t\t\t'
                matrix_str += row_str + '\n'
            return matrix_str

def save_summary(model,scores,u_label_counts,cr,mtx):

    label_c = tabulate(u_label_counts, headers=['Class', 'Count'], tablefmt='grid')

    with open('output.txt', 'w') as f:
        f.write("Data Analysis:\n")
        f.write(f'Training data shape: {X_train.shape}, {y_train.shape}\n')
        f.write(f'Testing data shape: {X_test.shape}, {y_test.shape}\n')
        f.write("Distribution of Data in Training Dataset: \n")
        f.write(label_c)
        f.write("\n")
        f.write("For training image samples check plots/Train_images.png \n\n")
        f.write("Model Architecture:\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f"\nEvaluation Accuracy: {scores} %\n")
        f.write("Classifiaction Report:\n")
        f.write(cr)
        f.write("Confusion Matrix:\n")
        f.write(mtx)

        # Add additional insights or observations if necessary


def main(data_folder):
    try:
        # Load and preprocess data
        #X_train, y_train, X_test, y_test = 
        load_data(data_folder)

        data_analysis()

        u_label_counts = []

        # Print unique values and their counts and add them to the list
        for label, count in zip(unique_labels, label_counts):
            u_label_counts.append((labels[label], count))
       
        model = load_model('final_model.keras')
        accuracy, loss= evaluate_model(model)
        
        accuracy = str(accuracy*100) + "%"
        print(accuracy)
        #predictions = model.predict(X_test)
        Y_pred = model.predict(X_test)
        # Convert predictions classes to one hot vectors 
        Y_pred_classes = np.argmax(Y_pred,axis = 1) 
        # Convert validation observations to one hot vectors
        Y_true = np.argmax(y_test,axis = 1) 
        # compute the confusion matrix
        confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
        # plot the confusion matrix
        plot_confusion_matrix(confusion_mtx,class_names)
        confusion_table = tabulate(confusion_mtx, headers=class_names, showindex=class_names, tablefmt='grid')
        print(confusion_table)
        cr=classification_report(Y_true, Y_pred_classes, target_names = class_names)


        save_summary(model,accuracy,u_label_counts,cr,confusion_table)

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
