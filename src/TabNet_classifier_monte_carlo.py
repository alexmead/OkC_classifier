"""
Author: Alex R. Mead
Date: August 2024

Description:
Noticing the Logistic Regression and Deep Learning approach both 
perform about the same after a single training cycle, here we 
explore the nature of each approach with a statistical analysis
to see if in fact the Deep Learning approach does better, or if 
it is mearly a "fluke" of the data. 

"""

import pandas as pd

##################################################################
# Load the dataset
df = pd.read_csv("data/okcupid_profiles.csv")
# Drop essay data
essay_cols = ['essay'+str(num) for num in range(0,10)]
df = df.drop(essay_cols, axis=1)
df = df.dropna() # Start simple, just drop all non-complete rows'
# Note: Much thought can be put into data "backfilling".

##################################################################
# A nice little preprocess of the data type: religion
# from src.utils import parse_religion_importance, parse_religion_type
# df['religious_importance'] = df['religion'].map(parse_religion_importance)
# df['religion'] = df['religion'].map(parse_religion_type) # override existing field 

##################################################################
# Process the indenpendant variables 
# Begin to organize the data and perform true categorical preprocessing
target = 'sex'
num_features = [
        # 'age', 
        'height', 
        # 'income',
        ]
cat_features = [
        'body_type', 
        # 'diet', 
        # 'drinks', 
        # 'drugs', 
        # 'education', 
        # 'ethnicity', 
        # 'job', 
        # 'last_online', 
        # 'location', 
        # 'offspring', 
        # 'orientation', 
        # 'pets', 
        # 'religion',
        # 'religious_importance',
        # 'sign', 
        # 'smokes', 
        # 'status',
        # 'speaks',
    ] # categorical features
features = num_features + cat_features
# Remove unused column
cols = list(set(list(df.columns)) - (set(features) | set([target])))
df.drop(cols, axis=1, inplace=True)

##############################
# One-Hot-Encoding 'manually'
df = pd.get_dummies(df, columns=['body_type']) 
df.drop('body_type_average', axis=1, inplace=True) # drugs=never is base case

# Encoding the data to be put into a Deep Neural Network (DNN):
# I'm not entirely sure on this step beyond a data typing issue.
from sklearn.preprocessing import LabelEncoder

nunique = df.nunique()
types = df.dtypes

categorical_columns = []
categorical_dims =  {}
for col in df.columns:
    if types[col] == 'object' or nunique[col] < 200:
        print(col, df[col].nunique())
        l_enc = LabelEncoder()
        df[col] = df[col].fillna("VV_likely")
        df[col] = l_enc.fit_transform(df[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        pass

##################################################################
# Extract observed, predicted (independant, dependant) variables
# gender = df.sex.values
gender = df.sex
df.drop(['sex'], axis=1, inplace=True)

###################################################################################################################################
# Training Models is done below.
####################################################################################################################################

##################################################################
# create training, test data partitions
import time
from sklearn.model_selection import train_test_split
from src.utils import get_accuracy, plot_execution_times_histogram
from sklearn.linear_model import LogisticRegression

results = []
train_time = []
total_rounds = 1000
for i in range(0, total_rounds):
    t0 = int(time.time() * 1_000_000)
    # Notice, resample for each round.
    X_train, X_test, y_train, y_test = train_test_split(df, gender, test_size=0.2, stratify=gender)
    # Instantiate model
    log_clf = LogisticRegression(
        max_iter=200, # org=100
        random_state=16
    ) 
    # Train each model:
    rsp = log_clf.fit(X_train, y_train)
    y_pred = log_clf.predict(X_test)
    # Store iterations results for post analysis. 
    results.append(get_accuracy(log_clf, X_test, y_test, type='float'))
    train_time.append(int(time.time() * 1_000_000) - t0)
    if i % 25 == 0:
        print(f"Making progress: {i} / ")


plot_execution_times_histogram(results)
f = open('log_data.dat', 'w'); f.write(str(results)); f.close()
##################################################################

##################################################################
# Building the actual Deep Neural Network (DNN) for classification
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

results = []
train_time = []
total_rounds = 1000
for i in range(0, total_rounds):
    t0 = int(time.time() * 1_000_000)
    # Notice, resample for each round.
    X_train, X_test, y_train, y_test = train_test_split(df, gender, test_size=0.2, stratify=gender)
    
    # define the model
    clf= TabNetClassifier(optimizer_fn=torch.optim.Adam,
                        scheduler_params={"step_size":10, 
                                            "gamma":0.9},
                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                        )
    # fit the model 
    rsp = clf.fit(
        X_train.values,y_train,
        eval_set=[(X_train.values, y_train), (X_test.values, y_test)],
        eval_name=['train', 'test'],
        eval_metric=['auc','balanced_accuracy'],
        max_epochs=200, patience=20,
        batch_size=512, virtual_batch_size=512,
        num_workers=0,
        weights=1,
        drop_last=False
    )  
    # Store iterations results for post analysis. 
    results.append(get_accuracy(clf, X_test.values, y_test, type='float'))
    train_time.append(int(time.time() * 1_000_000) - t0)
    f = open('tabNet_data.dat', 'w'); f.write(str(results)); f.close()
    print(f"Making progress: {i} / ")


plot_execution_times_histogram(results)


##################################################################
# Bundle up the logistic regression, and others if we did those...
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

log_test_score = round(accuracy_score(y_test, log_clf.predict(X_test)) * 100,2)
log_accuracies = cross_val_score(estimator = log_clf, X = X_train, y = y_train, cv = 10)
log_train_score=round(log_accuracies.mean()*100,2)

results_df = pd.DataFrame(data=[["Logistic Regression", log_train_score, log_test_score],],
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df.index += 1 
results_df



