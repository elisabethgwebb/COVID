# COVID Severity Prediction 
#Machine Learning Final Project 
#Elisabeth Webb, Eric Wu, Ben DeLeon, and Calix Carrington

## Data: ##
The data can be accessed and read in using this link https://raw.githubusercontent.com/elisabethgwebb/COVID/master/covid_sym.csv

It is also located in the COVID folder. 

## Libraries to Import: ##

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

from sklearn import preprocessing, metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split 

from sklearn.svm import SVC 

from sklearn.model_selection import cross_val_score,cross_val_predict

from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import LabelBinarizer

## Instructions to Run: ##

Run each section in the file named COVID_ver3.ipynb in the COVID folder. Each section in the notebook is labelled, and there are comments explaining the code. 

## Method: ##

First we read in the data and do exploratory data analysis to understand the data better. We looked at the missing values as the proportions of variables. Next, we did pre-processing in order to transform the severity levels into a binary variable. Then we begin building a logistic regression model and identify the feature importance of a severe case. After that we built a random forest model, and then compared multiple algorithms' accuracy scores in order to determine the best one. We then use clustering as a supervised model to identify the number of clusters, which is two that represent the target variable. Then we compare the true label and clustered label to calculate the accuracy. After seeing that Random Forest had the highest accuracy score, we tuned that model by altering the hyperparameters with the goal of improving the model. We evaluated the model with the best hyperparameters and then graphed the AUC curve and calculated the score. Lastly, we used the random forest predictive model to predict the severity level of 5 new patients and got the results. 

## Summary: ##

The goal of this project was to use what we have learned in our machine learning class to answer a predictive question about Coronavirus. Our question was to predict an incoming Coronavirus patient's severity based on their symptoms. We answered this question by building a random forest model that was able to predict a patient's severity with 87% accuracy. We also found the characteristics that contribute most to critical and deceased cases using logistic regression. 

