# Santiago Matallana
# Machine Learning for Public Policy - Assignment 3


from __future__ import division
from sklearn.metrics import *
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
from scipy import optimize
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os

plt.rcParams["figure.figsize"] = [20.0, 20.0]

# 1. Read Data: Assume input is CSV

def read_data(filename):
    '''
    Takes a filename and returns a pandas dataframe.
    '''
    original = pd.read_csv(filename, header=0)
    df = original.copy()
    return df    


# 2. Explore data

def explore_data(dataframe, histograms=True, scattermatrix=True, export_summary=True):
    '''
    Given a dataframe, outputs a csv summary, histograms, and a scatter matrix.
    Returns summary table.
    '''
    # Create directory for output
    from pandas.tools.plotting import scatter_matrix
    if not os.path.exists('output'):
        os.mkdir('output')
    print('Descriptive statistics exported as "output/summary.csv"')
    summary = dataframe.describe(include='all').round(2).transpose()
    summary.insert(1, 'missing', len(dataframe) - summary['count'])
    summary.insert(0, 'type', dataframe.dtypes)
    if export_summary == True:
        summary.to_csv('output/summary.csv')
    if histograms == True:
        print('Histograms exported as "output/histograms.png"')
        dataframe.hist();
        plt.savefig('output/histograms.png')
    if scattermatrix == True:
        print('Scatter matrix exported as "output/scatter_matrix.png"')
        scatter_matrix(dataframe, diagonal='kde');
        plt.savefig('output/scatter_matrix.png')
    return summary


# 3. Pre-process data

def split_train_test(dataframe, features, outcome_var, test_size=0.2):
    '''
    Splits dataframe into training and test sets, given test size
    '''
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(dataframe[features], 
        dataframe[outcome_var], test_size=test_size, random_state=0)
    
    return X_train, X_test, y_train, y_test

def impute_missing_train(dataframe, missing_values='NaN', strategy='mean'):
    '''
    Given a dataframe, imputes missing values with a given strategy.
    Supported strategies: 'mean', 'median', 'most_frequent'.
    Returns dictionary mapping transformed columns to its imputer value.
    '''
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values=missing_values, strategy=strategy, axis=0)
    imputed = imp.fit_transform(dataframe)
    df = pd.DataFrame(imputed)
    df.columns = list(dataframe.columns)
    
    imputers = {}
    if strategy == 'mean':
        for col in df.columns:
            mean = df[col].mean()
            imputers[col] = mean
    if strategy == 'median':
        for col in df.columns:
            median = df[col].median()
            imputers[col] = median
    if strategy == 'most_frequent':
        for col in df.columns:
            mode = df[col].mode()
            imputers[col] = mode
    return df, imputers

def impute_missing_test(dataframe, imputers):
    '''
    Uses train set imputers to fill in missing values on test set
    '''
    for col in imputers.keys():
        dataframe[col].fillna(imputers[col], inplace=True)

# 4. Generate features

# "Write a sample function that can discretize a continuous variable and one
# function that can take a categorical variable and create binary variables 
# from it."

def discretize(dataframe, variable, num_bins, labels=None):
    '''
    Discretizes a continuous variable based on specified number of bins
    '''
    new_name = variable + '_bins'
    dataframe[new_name] = pd.cut(dataframe[variable], bins=num_bins, labels=labels)

def binarize(dataframe, column):
    '''
    Takes a categorical variable (series), creates binary variables from it, and 
    appends them to dataframe.
    '''
    dummies = pd.get_dummies(column)
    for col_name in dummies.columns:
        dataframe[column.name + "=" + str(col_name)] = dummies[col_name]
    return dataframe

def cap(x, cap):
    '''
    Caps to adjust for outliers
    '''
    if x < cap:
        return x
    else:
        return cap

def func_to_feature(dataframe, feature, f):
    '''
    Applies a given function to a given feature, and adds corresponding column
    '''
    f_feat = 'f({})'.format(feature)
    dataframe[f_feat] = dataframe[feature].apply(lambda x: f(x))

# 5. Build classifiers

def define_clfs_params():

    clfs = {
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3)
        }
        
    grid = { 
        #'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'RF':{'n_estimators': [1,10], 'max_depth': [1,5], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5]},
        #'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
        'LR': {'penalty': ['l1','l2'], 'C': [0.00001,0.0001]},
        #'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
        'SGD': {'loss': ['hinge'], 'penalty': ['l2','l1']},
        #'ET': {'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'ET': {'n_estimators': [1,10], 'criterion': ['gini', 'entropy'],'max_depth': [1,5], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5]},
        #'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
        'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10]},
        #'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
        'GB': {'n_estimators': [1,10], 'learning_rate' : [0.001,0.01],'subsample' : [0.1,0.5], 'max_depth': [1,3]},
        'NB' : {},
        #'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5]},
        #'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
        'SVM' :{'C' :[0.00001,0.0001],'kernel':['linear']},
        #'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
        'KNN' :{'n_neighbors': [1,5],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree']}
        }

    return clfs, grid


def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test):
    '''
    '''
    rv = pd.DataFrame(columns=['classifier','precision', 'accuracy', 'recall', 'f1', 'aucurve', 'prcurve'])

    for index, clf in enumerate([clfs[x] for x in models_to_run]):
        model_running = models_to_run[index]
        parameter_values = grid[model_running]
        
        for p in ParameterGrid(parameter_values):

            try:
                clf.set_params(**p)
                print(clf)
                y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                precision, accuracy, recall, f1, aucurve, prcurve = precision_at_k(y_test, y_pred, 0.05)
                print('precision: ', precision)
                print('accuracy: ', accuracy)
                print('f1: ', f1)
                print('aucurve: ', aucurve)
                print('prcurve: ', prcurve)
                rv.loc[len(rv)] = pd.Series({'classifier':clf, 'precision':precision, 'accuracy':accuracy, 'recall':recall, 'f1':f1, 'aucurve':aucurve, 'prcurve':prcurve})
            except (IndexError):
                print('Error')
                continue

                # start_time = time.time()

    return rv

def precision_at_k(y_test, y_pred, k):
    threshold = np.sort(y_pred)[:: -1][int(k * len(y_pred))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_pred])
    precision = metrics.precision_score(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    aucurve = metrics.roc_auc_score(y_test, y_pred)
    prcurve = metrics.precision_recall_curve(y_test, y_pred)
    return precision, accuracy, recall, f1, aucurve, prcurve

