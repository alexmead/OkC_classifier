"""
Author: Alex R. Mead
Date: August 2024

Description:
Holds utility functions which are used by various scripts
within the src/ directory. They are grouped here for 
organizational and "cleaness" of the scripts in which 
they are used. 

"""



def parse_religion_importance(religion):
    """Returns the importance part of the religion part."""
    importance_types = [
        'very serious about it', 
        'somewhat serious about it', 
        'not too serious about it',
        'laughing about it'
        ]
    # Loop over importance values, return if included, else, "none"
    for importance in importance_types:
        if importance in religion:
            return importance
    return "none"


def parse_religion_type(religion):
    """Returns the type part of the religion part."""
    religion_types = [
        'agnosticism', 
        'atheism',
        'buddhism',
        'catholicism',
        'christianity', 
        'hinduism',
        'islam',
        'judaism',
        'other',
        ]
    # Loop over importance values, return if included, else, "none"
    for type_ in religion_types:
        if type_ in religion:
            return type_
    return "none"


import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

def print_score(clf, X_train, y_train, X_test, y_test, score, train=True):
     """Taken from kaggle: https://www.kaggle.com/code/enigmak/tabnet-deep-neural-network-for-tabular-data"""
     if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        accuracies = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 10)
        print("Train Result:\n================================================")
        print(clf,":Accuracy: {:.2f} %".format(accuracies.mean()*100))
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        score.append(accuracies.mean()*100)
     elif train==False:
         pred = clf.predict(X_test)
         clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
         print("Test Result:\n================================================")        
         print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
         print("_______________________________________________")
         print(f"CLASSIFICATION REPORT:\n{clf_report}")
         print("_______________________________________________")
         print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
         return f"{accuracy_score(y_test, pred) * 100:.2f}"

def get_accuracy(clf, X_train, y_test, type='float'):
    """Function to return the accuracy of the trained model."""
    pred = clf.predict(X_train)
    acc_str = f"{accuracy_score(y_test, pred) * 100:.2f}" # use this weird string thing to get 2 decimals is all.
    type_casted = None
    match type:
        case 'float':
            type_casted = float(acc_str)
        case 'string':
            type_casted = str(acc_str)
        case 'integer':
            type_casted = int(acc_str)
    return type_casted


import matplotlib.pyplot as plt
import numpy as np

def plot_execution_times_histogram(execution_times):
    # Define the bins, rounded to 1 decimal place width
    bins = np.arange(min(execution_times), max(execution_times) + 0.1, 0.1)
    # Plot histogram
    plt.hist(execution_times, bins=bins, edgecolor='black', density=False)  # density=True for PDF
    plt.xlabel('Model Prediction Accuracy [%]')
    plt.ylabel('Number of Occurances')
    plt.title('Histogram of Model Accuracy')
    plt.show()
