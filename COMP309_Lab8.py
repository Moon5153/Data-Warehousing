# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 19:46:27 2022

@author: Kokila
"""

# Importing the data set iris
import pandas as pd
import os
path = "C:/Users/Kokila/Downloads"
filename = 'iris1.csv'
fullpath = os.path.join(path,filename)
data_kokila_i = pd.read_csv(fullpath,sep=',')
print(data_kokila_i)
print(data_kokila_i.columns.values)
print(data_kokila_i.shape)
print(data_kokila_i.describe())
print(data_kokila_i.dtypes)
print(data_kokila_i.head(5))
print(data_kokila_i['Species'].unique())


#Splitting the predictor and target variables
colnames=data_kokila_i.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]
print(target)
# splitting the dataset into train and test variables
import numpy as np
data_kokila_i['is_train'] = np.random.uniform(0, 1, len(data_kokila_i)) <= .75
print(data_kokila_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_kokila_i[data_kokila_i['is_train']==True], data_kokila_i[data_kokila_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

from sklearn.tree import DecisionTreeClassifier
dt_kokila = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_kokila.fit(train[predictors], train[target])

preds=dt_kokila.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])

from sklearn.tree import export_graphviz
with open('C:/Users/Kokila/Downloads/dtree2.dot', 'w') as dotfile:
    export_graphviz(dt_kokila, out_file = dotfile, feature_names = predictors)
dotfile.close()

X=data_kokila_i[predictors]
Y=data_kokila_i[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)

dt1_kokila = DecisionTreeClassifier(criterion='entropy',max_depth=5, min_samples_split=20,
random_state=99)
dt1_kokila.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_kokila, trainX, trainY, scoring='accuracy', cv=crossvalidation,
n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1_kokila.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=None))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(testY, testY_predict, labels=None)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa',
'versicolor', 'virginica']);
plt.show()


# Importing the data set iris
import pandas as pd
import os
path = "C:/Users/Kokila/Downloads"
filename = 'iris1.csv'
fullpath = os.path.join(path,filename)
data_kokila_i = pd.read_csv(fullpath,sep=',')
print(data_kokila_i)
print(data_kokila_i.columns.values)
print(data_kokila_i.shape)
print(data_kokila_i.describe())
print(data_kokila_i.dtypes)
print(data_kokila_i.head(5))
print(data_kokila_i['Species'].unique())


#Splitting the predictor and target variables
colnames=data_kokila_i.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]
print(target)
# splitting the dataset into train and test variables
import numpy as np
data_kokila_i['is_train'] = np.random.uniform(0, 1, len(data_kokila_i)) <= .75
print(data_kokila_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_kokila_i[data_kokila_i['is_train']==True], data_kokila_i[data_kokila_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

from sklearn.tree import DecisionTreeClassifier
dt_kokila = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_kokila.fit(train[predictors], train[target])

preds=dt_kokila.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])

from sklearn.tree import export_graphviz
with open('C:/Users/Kokila/Downloads/dtree2.dot', 'w') as dotfile:
    export_graphviz(dt_kokila, out_file = dotfile, feature_names = predictors)
dotfile.close()

X=data_kokila_i[predictors]
Y=data_kokila_i[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
dt1_kokila = DecisionTreeClassifier(criterion='entropy',max_depth=1, min_samples_split=20,
random_state=99)
dt1_kokila.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_kokila, trainX, trainY, scoring='accuracy', cv=crossvalidation,
n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1_kokila.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=None))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(testY, testY_predict, labels=None)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa',
'versicolor', 'virginica']);
plt.show()

# Importing the data set iris
import pandas as pd
import os
path = "C:/Users/Kokila/Downloads"
filename = 'iris1.csv'
fullpath = os.path.join(path,filename)
data_kokila_i = pd.read_csv(fullpath,sep=',')
print(data_kokila_i)
print(data_kokila_i.columns.values)
print(data_kokila_i.shape)
print(data_kokila_i.describe())
print(data_kokila_i.dtypes)
print(data_kokila_i.head(5))
print(data_kokila_i['Species'].unique())


#Splitting the predictor and target variables
colnames=data_kokila_i.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]
print(target)
# splitting the dataset into train and test variables
import numpy as np
data_kokila_i['is_train'] = np.random.uniform(0, 1, len(data_kokila_i)) <= .75
print(data_kokila_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_kokila_i[data_kokila_i['is_train']==True], data_kokila_i[data_kokila_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

from sklearn.tree import DecisionTreeClassifier
dt_kokila = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_kokila.fit(train[predictors], train[target])

preds=dt_kokila.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])

from sklearn.tree import export_graphviz
with open('C:/Users/Kokila/Downloads/dtree2.dot', 'w') as dotfile:
    export_graphviz(dt_kokila, out_file = dotfile, feature_names = predictors)
dotfile.close()

X=data_kokila_i[predictors]
Y=data_kokila_i[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
dt1_kokila = DecisionTreeClassifier(criterion='entropy',max_depth=2, min_samples_split=20,
random_state=99)
dt1_kokila.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_kokila, trainX, trainY, scoring='accuracy', cv=crossvalidation,
n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1_kokila.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=None))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(testY, testY_predict, labels=None)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa',
'versicolor', 'virginica']);
plt.show()


# Importing the data set iris
import pandas as pd
import os
path = "C:/Users/Kokila/Downloads"
filename = 'iris1.csv'
fullpath = os.path.join(path,filename)
data_kokila_i = pd.read_csv(fullpath,sep=',')
print(data_kokila_i)
print(data_kokila_i.columns.values)
print(data_kokila_i.shape)
print(data_kokila_i.describe())
print(data_kokila_i.dtypes)
print(data_kokila_i.head(5))
print(data_kokila_i['Species'].unique())


#Splitting the predictor and target variables
colnames=data_kokila_i.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]
print(target)
# splitting the dataset into train and test variables
import numpy as np
data_kokila_i['is_train'] = np.random.uniform(0, 1, len(data_kokila_i)) <= .75
print(data_kokila_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_kokila_i[data_kokila_i['is_train']==True], data_kokila_i[data_kokila_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

from sklearn.tree import DecisionTreeClassifier
dt_kokila = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_kokila.fit(train[predictors], train[target])

preds=dt_kokila.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])

from sklearn.tree import export_graphviz
with open('C:/Users/Kokila/Downloads/dtree2.dot', 'w') as dotfile:
    export_graphviz(dt_kokila, out_file = dotfile, feature_names = predictors)
dotfile.close()

X=data_kokila_i[predictors]
Y=data_kokila_i[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
dt1_kokila = DecisionTreeClassifier(criterion='entropy',max_depth=3, min_samples_split=20,
random_state=99)
dt1_kokila.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_kokila, trainX, trainY, scoring='accuracy', cv=crossvalidation,
n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1_kokila.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=None))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(testY, testY_predict, labels=None)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa',
'versicolor', 'virginica']);
plt.show()


# Importing the data set iris
import pandas as pd
import os
path = "C:/Users/Kokila/Downloads"
filename = 'iris1.csv'
fullpath = os.path.join(path,filename)
data_kokila_i = pd.read_csv(fullpath,sep=',')
print(data_kokila_i)
print(data_kokila_i.columns.values)
print(data_kokila_i.shape)
print(data_kokila_i.describe())
print(data_kokila_i.dtypes)
print(data_kokila_i.head(5))
print(data_kokila_i['Species'].unique())


#Splitting the predictor and target variables
colnames=data_kokila_i.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]
print(target)
# splitting the dataset into train and test variables
import numpy as np
data_kokila_i['is_train'] = np.random.uniform(0, 1, len(data_kokila_i)) <= .75
print(data_kokila_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_kokila_i[data_kokila_i['is_train']==True], data_kokila_i[data_kokila_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

from sklearn.tree import DecisionTreeClassifier
dt_kokila = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_kokila.fit(train[predictors], train[target])

preds=dt_kokila.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])

from sklearn.tree import export_graphviz
with open('C:/Users/Kokila/Downloads/dtree2.dot', 'w') as dotfile:
    export_graphviz(dt_kokila, out_file = dotfile, feature_names = predictors)
dotfile.close()

X=data_kokila_i[predictors]
Y=data_kokila_i[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
dt1_kokila = DecisionTreeClassifier(criterion='entropy',max_depth=4, min_samples_split=20,
random_state=99)
dt1_kokila.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_kokila, trainX, trainY, scoring='accuracy', cv=crossvalidation,
n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1_kokila.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=None))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(testY, testY_predict, labels=None)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa',
'versicolor', 'virginica']);
plt.show()


# Importing the data set iris
import pandas as pd
import os
path = "C:/Users/Kokila/Downloads"
filename = 'iris1.csv'
fullpath = os.path.join(path,filename)
data_kokila_i = pd.read_csv(fullpath,sep=',')
print(data_kokila_i)
print(data_kokila_i.columns.values)
print(data_kokila_i.shape)
print(data_kokila_i.describe())
print(data_kokila_i.dtypes)
print(data_kokila_i.head(5))
print(data_kokila_i['Species'].unique())


#Splitting the predictor and target variables
colnames=data_kokila_i.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]
print(target)
# splitting the dataset into train and test variables
import numpy as np
data_kokila_i['is_train'] = np.random.uniform(0, 1, len(data_kokila_i)) <= .75
print(data_kokila_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_kokila_i[data_kokila_i['is_train']==True], data_kokila_i[data_kokila_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

from sklearn.tree import DecisionTreeClassifier
dt_kokila = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_kokila.fit(train[predictors], train[target])

preds=dt_kokila.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])

from sklearn.tree import export_graphviz
with open('C:/Users/Kokila/Downloads/dtree2.dot', 'w') as dotfile:
    export_graphviz(dt_kokila, out_file = dotfile, feature_names = predictors)
dotfile.close()

X=data_kokila_i[predictors]
Y=data_kokila_i[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
dt1_kokila = DecisionTreeClassifier(criterion='entropy',max_depth=5, min_samples_split=20,
random_state=99)
dt1_kokila.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_kokila, trainX, trainY, scoring='accuracy', cv=crossvalidation,
n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1_kokila.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=None))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(testY, testY_predict, labels=None)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa',
'versicolor', 'virginica']);
plt.show()


# Importing the data set iris
import pandas as pd
import os
path = "C:/Users/Kokila/Downloads"
filename = 'iris1.csv'
fullpath = os.path.join(path,filename)
data_kokila_i = pd.read_csv(fullpath,sep=',')
print(data_kokila_i)
print(data_kokila_i.columns.values)
print(data_kokila_i.shape)
print(data_kokila_i.describe())
print(data_kokila_i.dtypes)
print(data_kokila_i.head(5))
print(data_kokila_i['Species'].unique())


#Splitting the predictor and target variables
colnames=data_kokila_i.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]
print(target)
# splitting the dataset into train and test variables
import numpy as np
data_kokila_i['is_train'] = np.random.uniform(0, 1, len(data_kokila_i)) <= .75
print(data_kokila_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_kokila_i[data_kokila_i['is_train']==True], data_kokila_i[data_kokila_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

from sklearn.tree import DecisionTreeClassifier
dt_kokila = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_kokila.fit(train[predictors], train[target])

preds=dt_kokila.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])

from sklearn.tree import export_graphviz
with open('C:/Users/Kokila/Downloads/dtree2.dot', 'w') as dotfile:
    export_graphviz(dt_kokila, out_file = dotfile, feature_names = predictors)
dotfile.close()

X=data_kokila_i[predictors]
Y=data_kokila_i[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
dt1_kokila = DecisionTreeClassifier(criterion='entropy',max_depth=6, min_samples_split=20,
random_state=99)
dt1_kokila.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_kokila, trainX, trainY, scoring='accuracy', cv=crossvalidation,
n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1_kokila.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=None))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(testY, testY_predict, labels=None)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa',
'versicolor', 'virginica']);
plt.show()


# Importing the data set iris
import pandas as pd
import os
path = "C:/Users/Kokila/Downloads"
filename = 'iris1.csv'
fullpath = os.path.join(path,filename)
data_kokila_i = pd.read_csv(fullpath,sep=',')
print(data_kokila_i)
print(data_kokila_i.columns.values)
print(data_kokila_i.shape)
print(data_kokila_i.describe())
print(data_kokila_i.dtypes)
print(data_kokila_i.head(5))
print(data_kokila_i['Species'].unique())


#Splitting the predictor and target variables
colnames=data_kokila_i.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]
print(target)
# splitting the dataset into train and test variables
import numpy as np
data_kokila_i['is_train'] = np.random.uniform(0, 1, len(data_kokila_i)) <= .75
print(data_kokila_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_kokila_i[data_kokila_i['is_train']==True], data_kokila_i[data_kokila_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

from sklearn.tree import DecisionTreeClassifier
dt_kokila = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_kokila.fit(train[predictors], train[target])

preds=dt_kokila.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])

from sklearn.tree import export_graphviz
with open('C:/Users/Kokila/Downloads/dtree2.dot', 'w') as dotfile:
    export_graphviz(dt_kokila, out_file = dotfile, feature_names = predictors)
dotfile.close()

X=data_kokila_i[predictors]
Y=data_kokila_i[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
dt1_kokila = DecisionTreeClassifier(criterion='entropy',max_depth=7, min_samples_split=20,
random_state=99)
dt1_kokila.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_kokila, trainX, trainY, scoring='accuracy', cv=crossvalidation,
n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1_kokila.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=None))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(testY, testY_predict, labels=None)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa',
'versicolor', 'virginica']);
plt.show()

# Importing the data set iris
import pandas as pd
import os
path = "C:/Users/Kokila/Downloads"
filename = 'iris1.csv'
fullpath = os.path.join(path,filename)
data_kokila_i = pd.read_csv(fullpath,sep=',')
print(data_kokila_i)
print(data_kokila_i.columns.values)
print(data_kokila_i.shape)
print(data_kokila_i.describe())
print(data_kokila_i.dtypes)
print(data_kokila_i.head(5))
print(data_kokila_i['Species'].unique())


#Splitting the predictor and target variables
colnames=data_kokila_i.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]
print(target)
# splitting the dataset into train and test variables
import numpy as np
data_kokila_i['is_train'] = np.random.uniform(0, 1, len(data_kokila_i)) <= .75
print(data_kokila_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_kokila_i[data_kokila_i['is_train']==True], data_kokila_i[data_kokila_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

from sklearn.tree import DecisionTreeClassifier
dt_kokila = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_kokila.fit(train[predictors], train[target])

preds=dt_kokila.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])

from sklearn.tree import export_graphviz
with open('C:/Users/Kokila/Downloads/dtree2.dot', 'w') as dotfile:
    export_graphviz(dt_kokila, out_file = dotfile, feature_names = predictors)
dotfile.close()

X=data_kokila_i[predictors]
Y=data_kokila_i[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
dt1_kokila = DecisionTreeClassifier(criterion='entropy',max_depth=8, min_samples_split=20,
random_state=99)
dt1_kokila.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_kokila, trainX, trainY, scoring='accuracy', cv=crossvalidation,
n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1_kokila.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=None))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(testY, testY_predict, labels=None)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa',
'versicolor', 'virginica']);
plt.show()


# Importing the data set iris
import pandas as pd
import os
path = "C:/Users/Kokila/Downloads"
filename = 'iris1.csv'
fullpath = os.path.join(path,filename)
data_kokila_i = pd.read_csv(fullpath,sep=',')
print(data_kokila_i)
print(data_kokila_i.columns.values)
print(data_kokila_i.shape)
print(data_kokila_i.describe())
print(data_kokila_i.dtypes)
print(data_kokila_i.head(5))
print(data_kokila_i['Species'].unique())


#Splitting the predictor and target variables
colnames=data_kokila_i.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]
print(target)
# splitting the dataset into train and test variables
import numpy as np
data_kokila_i['is_train'] = np.random.uniform(0, 1, len(data_kokila_i)) <= .75
print(data_kokila_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_kokila_i[data_kokila_i['is_train']==True], data_kokila_i[data_kokila_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

from sklearn.tree import DecisionTreeClassifier
dt_kokila = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_kokila.fit(train[predictors], train[target])

preds=dt_kokila.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])

from sklearn.tree import export_graphviz
with open('C:/Users/Kokila/Downloads/dtree2.dot', 'w') as dotfile:
    export_graphviz(dt_kokila, out_file = dotfile, feature_names = predictors)
dotfile.close()

X=data_kokila_i[predictors]
Y=data_kokila_i[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
dt1_kokila = DecisionTreeClassifier(criterion='entropy',max_depth=9, min_samples_split=20,
random_state=99)
dt1_kokila.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_kokila, trainX, trainY, scoring='accuracy', cv=crossvalidation,
n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1_kokila.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=None))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(testY, testY_predict, labels=None)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa',
'versicolor', 'virginica']);
plt.show()

# Importing the data set iris
import pandas as pd
import os
path = "C:/Users/Kokila/Downloads"
filename = 'iris1.csv'
fullpath = os.path.join(path,filename)
data_kokila_i = pd.read_csv(fullpath,sep=',')
print(data_kokila_i)
print(data_kokila_i.columns.values)
print(data_kokila_i.shape)
print(data_kokila_i.describe())
print(data_kokila_i.dtypes)
print(data_kokila_i.head(5))
print(data_kokila_i['Species'].unique())


#Splitting the predictor and target variables
colnames=data_kokila_i.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]
print(target)
# splitting the dataset into train and test variables
import numpy as np
data_kokila_i['is_train'] = np.random.uniform(0, 1, len(data_kokila_i)) <= .75
print(data_kokila_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_kokila_i[data_kokila_i['is_train']==True], data_kokila_i[data_kokila_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

from sklearn.tree import DecisionTreeClassifier
dt_kokila = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_kokila.fit(train[predictors], train[target])

preds=dt_kokila.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])

from sklearn.tree import export_graphviz
with open('C:/Users/Kokila/Downloads/dtree2.dot', 'w') as dotfile:
    export_graphviz(dt_kokila, out_file = dotfile, feature_names = predictors)
dotfile.close()

X=data_kokila_i[predictors]
Y=data_kokila_i[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
dt1_kokila = DecisionTreeClassifier(criterion='entropy',max_depth=10, min_samples_split=20,
random_state=99)
dt1_kokila.fit(trainX,trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt1_kokila, trainX, trainY, scoring='accuracy', cv=crossvalidation,
n_jobs=1))
score

### Test the model using the testing data
testY_predict = dt1_kokila.predict(testX)
testY_predict.dtype
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = Y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=None))

import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(testY, testY_predict, labels=None)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa',
'versicolor', 'virginica']);
plt.show()



# decision tree for feature importance 

from matplotlib import pyplot

# get importance
importance = dt_kokila.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()