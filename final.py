#!/usr/bin/python3

###
# Computer and Information Science
# CIS 472 taught by Thien Huu Nguyen
# Final Project Code
# Author: Irfan Filipovic
###

# League of legends blue win classifiers

import numpy as np
import re
import sklearn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression

# read_data used from assignment 4 for CIS 472
def read_data(filename):
  # Read names
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  f.close()
  # Read data
  data = np.genfromtxt(filename, delimiter=',', skip_header=1)
  # 2:21 is all blue variables, can allow red if change 21 to empty (i.e [:, 2:21] -> [:, 2:])
  # Using only blue variables allows for user friendly small data predictor, realistically not enough time for a player to check enemy data as well
  print(data.shape)
  x = data[:, 2:21]
  y = data[:, 1]
  varnames = varnames[:21]
  return ((x,y), varnames)

# Function to split data into three sets
def split_data(data_x, data_y):
    length = len(data_y)
    # 70 percent training data
    trainingInd = (int(0), int(round(length * .7)))
    # 20 percent testing data
    testingInd = (trainingInd[1] + 1, int(trainingInd[1] + round((length * .2))))
    # 10 percent held out data
    heldInd = (testingInd[1] + 1, int(testingInd[1] + round((length * .1))))
    training_x = data_x[trainingInd[0]:trainingInd[1]]
    training_y = data_y[trainingInd[0]:trainingInd[1]]
    testing_x = data_x[testingInd[0]:testingInd[1]]
    testing_y = data_y[testingInd[0]:testingInd[1]]
    held_x = data_x[heldInd[0]:heldInd[1]]
    held_y = data_y[heldInd[0]:heldInd[1]]
    return training_x, training_y, testing_x, testing_y, held_x, held_y

def smaller_data(data_x):
    # Readily available data

    # first blood [2], kills [3], deaths [4], assists [5], elite[6], dragons [7], heralds [8], towers[9], total minions[13]
    # create array to return
    ret = []
    # for each example get the above variables and append to ret
    for i in range(len(data_x)):
        test = []
        test.append(data_x[i, 2])
        test.append(data_x[i, 3])
        test.append(data_x[i, 4])
        test.append(data_x[i, 5])
        test.append(data_x[i, 6])
        test.append(data_x[i, 7])
        test.append(data_x[i, 8])
        test.append(data_x[i, 9])
        test.append(data_x[i, 13])
        ret.append(test)
    # return smaller dataset
    return ret

def k_nearest(training_x, training_y, testing_x, testing_y, held_x, held_y):
    # K-neareset neighbor

    # K hyper params
    k_params = [1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11, 12, 13, 14, 15]

    # Create a model for each parameter
    models_K = []
    for k in k_params:
        model = neighbors.KNeighborsClassifier(n_neighbors= k, weights='uniform', algorithm='ball_tree')
        models_K.append(model)

    # Fit each model to training data
    for model in models_K:
        model.fit(training_x, training_y)

    # Variables to hold all model scores, best model and score
    scores_k = []
    best_k = 0
    main_k = None

    # Each model get a score from development set, if score highest then assign best
    for m in models_K:
        s = m.score(held_x, held_y)
        scores_k.append(s)
        if(best_k < s):
            best_k = s
            main_k = m
    # Score of each parameter
    print("Score of each param: ")
    print(scores_k) # used to see results of each parameter on development dataset
    # Predict using best parameter model
    k_pred = main_k.predict(testing_x)
    # As k_neighbors returns predicted outcome
    correct = 0
    for val in k_pred:
        if val == 1:
            correct += 1
    # compute accuracy with number correct over total number
    accuracy = (correct / len(k_pred))
    print("K_nearest accuracy: ", accuracy)

def log_regress(training_x, training_y, testing_x, testing_y, held_x, held_y):
    # Logisitic Regression using l2 regularizer

    # Logisitic parameters, learning rate
    lr_params = [1, .1, .01, .001, .0001]

    # Create a model for each parameter
    # max_iter was changed until convergence was reached when fitting models
    models_lr = []
    for lr in lr_params:
        model = LogisticRegression(penalty="l2", solver="saga", C=lr, max_iter=7000)
        models_lr.append(model)
    for lr in lr_params:
        model = LogisticRegression(penalty="l1", solver="saga", C=lr, max_iter=7000)
        models_lr.append(model)

    # Fit each model to training data
    for model in models_lr:
        model.fit(training_x, training_y)

    # Variables to hold all model scores, best model and score
    scores_lr = []
    best_lr = 0
    main_lr = None

    # Each model get a score from development set, if score highest then assign best
    for m in models_lr:
        s = m.score(held_x, held_y)
        scores_lr.append(s)
        if(best_lr < s):
            best_lr = s
            main_lr = m

    # Score of each parameter
    print("Score of each param: ")
    print(scores_lr) # used to see results of each parameter on development dataset
    # Predict using best parameter model
    lr_pred = main_lr.predict(testing_x)
    # As k_neighbors returns predicted outcome
    correct = 0
    for val in lr_pred:
        if val == 1:
            correct += 1
    # compute accuracy with number correct over total number
    accuracy = (correct / len(lr_pred))
    print("Logistic regression accuracy: ", accuracy)

def main():
    # Get reference to data
    dataPage = "high_diamond_ranked_10min.csv"
    # Read data into variables
    ((data_x, data_y), varnames) = read_data(dataPage)

    # Scale the data, set to integers
    preprocessing.scale(data_x)
    data_x = data_x.astype(int)

    # Split data into three datasets
    training_x, training_y, testing_x, testing_y, held_x, held_y = split_data(data_x, data_y)
    
    print("### Full Data ###")
    # Run K-nearest neighbors classifier
    k_nearest(training_x, training_y, testing_x, testing_y, held_x, held_y)
    
    # Run Logisitic Regression classifier
    log_regress(training_x, training_y, testing_x, testing_y, held_x, held_y)

    # Create dataset from variables a player may input while in game
    # Split smaller dataset into three datasets
    smalldata = smaller_data(data_x)
    smalltrain_x, smalltrain_y, smalltest_x, smalltest_y, smallheld_x, smallheld_y = split_data(smalldata, data_y)

    print("### Small Data ###")
    # Run K-nearest neighbors on smalldata
    k_nearest(smalltrain_x, smalltrain_y, smalltest_x, smalltest_y, smallheld_x, smallheld_y)

    # Run Logisitic Regression on smalldata
    log_regress(smalltrain_x, smalltrain_y, smalltest_x, smalltest_y, smallheld_x, smallheld_y)

if __name__ == "__main__":
    main()