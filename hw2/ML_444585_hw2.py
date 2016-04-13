
# coding: utf-8

# <div class="alert alert-success">
# **Author: Santiago Matallana**<br>
# Re: Machine Learning for Public Policy - Assignment 2<br>
# 2016-04-09
# </div>

# [**1. Read data**](#1.-Read-data)<br>
# [**2. Explore data**](#2.-Explore-data)<br>
# [**3. Pre-process data**](#3.-Pre-process-data)<br>
# [**4. Generate features**](#4.-Generate-features)<br>
# [**5. Build classifier**](#5.-Build-classifier)<br>
# [**6. Evaluate classifier**](#6.-Evaluate-classifier)

# In[90]:

#Requirements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams["figure.figsize"] = [20.0, 20.0]
plt.style.use('ggplot')


# <h1>1. Read data</h1>

# In[91]:

def read_data(filename):
    '''
    '''
    return pd.read_csv(filename)    


# ---

# In[92]:

train = read_data('data/cs-training.csv')
test = read_data('data/cs-test.csv')


# In[93]:

train.head()


# In[94]:

len(train)


# In[95]:

test.head()


# In[96]:

len(test)


# <h1>2. Explore data</h1>

# In[85]:

def explore_data(dataframe, histograms=True, scattermatrix=True, export_summary=True):
    '''
    '''
    # Create directory for output
    import os
    from pandas.tools.plotting import scatter_matrix
    if not os.path.exists('output'):
        os.mkdir('output')
    print('Descriptive statistics exported as "output/summary_original.csv"')
    summary = dataframe.describe(include='all').round(2).transpose()
    summary.insert(1, 'missing', len(dataframe) - summary['count'])
    summary.insert(0, 'type', dataframe.dtypes)
    if export_summary == True:
        summary.to_csv('output/summary_original.csv')
    if histograms == True:
        print('Histograms exported as "output/histograms.png"')
        dataframe.hist();
        plt.savefig('output/histograms.png')
    if scattermatrix == True:
        print('Scatter matrix exported as "output/scatter_matrix.png"')
        scatter_matrix(dataframe, diagonal='kde');
        plt.savefig('output/scatter_matrix.png')


# ---

# In[99]:

train.drop('Unnamed: 0', axis=1, inplace=True)


# In[108]:

test.drop('Unnamed: 0', axis=1, inplace=True)


# In[62]:

explore_data(train)


# <h1>3. Pre-process data</h1>

# In[100]:

def impute_missing(dataframe, missing_values='NaN', method='Imputer', strategy='mean'):
    '''
    '''
    if method == 'Imputer':
        from sklearn.preprocessing import Imputer
        imp = Imputer(missing_values=missing_values, strategy=strategy, axis=0)
        imputed = imp.fit_transform(dataframe)
        df = pd.DataFrame(imputed)
        df.columns = list(dataframe.columns)
#        print('Descriptive statistics of imputed dataframe exported as "output/summary_imputed.csv"')
#        summary = df.describe(include='all').round(2).transpose()
#        summary.insert(1, 'missing', len(df) - summary['count'])
#        summary.insert(0, 'type', df.dtypes)
#        summary.to_csv('output/summary_imputed.csv')
    return df


# ---

# In[111]:

outcome_var = 'SeriousDlqin2yrs'


# In[112]:

features = list(train.columns.difference([outcome_var]))
print(features)


# In[113]:

train = impute_missing(train)


# In[114]:

test = impute_missing(test[features])


# <h1>4. Generate features</h1>

# In[103]:

def cap(x, cap):
    '''
    Helper function to cap the value of a variable for discretization
    '''
    if x < cap:
        return x
    else:
        return cap

def discretize_continuous(dataframe, continuous_var, new_discrete_var, bins, capsize):
    '''
    Discretizes a continuous variable
    '''
    dataframe[new_discrete_var] = dataframe[continuous_var].apply(lambda x: cap(x, capsize))
    dataframe[new_discrete_var] = pd.cut(dataframe[new_discrete_var], bins=bins, labels=False)
    return dataframe


# In[104]:

def binary_from_categorial(dataframe, categorical_var):
    '''
    Takes a categorical variable, creates binary variables from it and appends them to dataframe
    '''
    df = pd.concat([dataframe, pd.get_dummies(dataframe[categorical_var])], axis=1)
    return df


# <h1>5. Build classifier</h1>

# In[105]:

def split_train_test(dataframe, features, outcome_var, test_size=0.2):
    '''
    Splits dataframe into training and test sets, given test size
    '''
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(dataframe[features], dataframe[outcome_var],                                                         test_size=test_size, random_state=0)
    return x_train, x_test, y_train, y_test    


# In[136]:

def classifier(dataframe, features, outcome_var, method='logit', test_size=0.2):
    '''
    '''
    if method == 'logit':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        return clf


# ---

# In[137]:

x_train, x_test, y_train, y_test = split_train_test(train, features, outcome_var)


# In[138]:

clf = classifier(train, features, outcome_var)


# <h1>6. Evaluate classifier</h1>

# In[143]:

def evaluate_clf(classifier=clf, metric='accuracy'):
    '''
    '''
    from sklearn.metrics import accuracy_score
    # Predict outcome on test set
    y_hat = clf.predict(x_test)
    if metric == 'accuracy':
        score = accuracy_score(y_test, y_hat)
        print('Your model has an accuracy of', '{:.2f}%.'.format(100 * score))


# ---

# In[144]:

evaluate_clf()


# # Predict

# In[164]:

def predict(dataframe, classifier=clf):
    y_hat = clf.predict(dataframe[features])
    dataframe['predicted'] = pd.Series(y_hat)
    print('Predicted dataframe exported as "output/predicted.csv"')
    dataframe.to_csv('output/predicted.csv')
    return dataframe


# In[166]:

test_predicted = predict(test)


# In[167]:

test_predicted['predicted'].value_counts()


# In[170]:

100 * 598 / (100905 + 598)

