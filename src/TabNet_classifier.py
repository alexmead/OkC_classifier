"""
Author: Alex R. Mead
Date: August 2024

Description:
Further exploring classification with tabular data related to 
the OkCupid Profiles dataset. Here, we use a Deep Learning
approach which is available as part of the TabNet module.

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

####################################################################################################################################
####################################################################################################################################
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
        
##############################

##################################################################
# Extract observed, predicted (independant, dependant) variables
gender = df.sex
df.drop(['sex'], axis=1, inplace=True)
##################################################################

####################################################################################################################################
####################################################################################################################################

##################################################################
# create training, test data partitions
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, gender, test_size=0.2, stratify=gender, random_state=1)
##################################################################

##################################################################
# Building the actual Deep Neural Network (DNN) for classification
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# define the model
clf= TabNetClassifier(optimizer_fn=torch.optim.Adam,
                       scheduler_params={"step_size":10, 
                                         "gamma":0.9},
                       scheduler_fn=torch.optim.lr_scheduler.StepLR,
                      )

# fit the model 
clf.fit(
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

from sklearn.metrics import classification_report
y_pred=clf.predict(X_test.values)
print(classification_report(y_test, y_pred, target_names=['male', 'female']))

correct_pred = (y_pred==y_test).sum().item()
accuracy = correct_pred / len(y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")


