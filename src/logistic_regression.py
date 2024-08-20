"""
Author: Alex R. Mead
Date: August 2024

Description:
Walks through the Logistic Regression approach to the
gender prediction question. Please find an accompying
blog post on my Substack at:
            

"""

import pandas as pd


##################################################################
# Load the dataset
df = pd.read_csv("data/okcupid_profiles.csv")
# Drop essay data
essay_cols = ['essay'+str(num) for num in range(0,10)]
df = df.drop(essay_cols, axis=1)
df = df.dropna() # Start simple, just drop all non-complete rows'
##################################################################
# Observed variables
obs_cols = [
    # 'age', 
    'body_type', 
    # 'diet', 
    # 'drinks', 
    # 'drugs', 
    # 'education', 
    # 'ethnicity', 
    'height', 
    # 'income', 
    # 'job', 
    # 'last_online', 
    # 'location', 
    # 'offspring', 
    # 'orientation', 
    # 'pets', 
    # 'religion', # Perhaps this is a place to explore variable "augmentation". This is really two variables (relgion, importance)
    # 'sign', 
    # 'smokes', 
    # 'status'
    ]
X = df[obs_cols]
# Further process "observed variables" for 
if 'drugs' in obs_cols:
    X = pd.get_dummies(X, columns=['drugs']) 
    X.drop('drugs_never', axis=1, inplace=True) # drugs=never is base case

# body_type
if 'body_type' in obs_cols:
    X = pd.get_dummies(X, columns=['body_type']) 
    X.drop('body_type_average', axis=1, inplace=True) # drugs=never is base case

# diet
if 'diet' in obs_cols:
    X = pd.get_dummies(X, columns=['diet']) 
    X.drop('diet_anything', axis=1, inplace=True) # drugs=never is base case

# drinks
if 'drinks' in obs_cols:
    X = pd.get_dummies(X, columns=['drinks']) 
    X.drop('drinks_not at all', axis=1, inplace=True) # drugs=never is base case

# income
if 'income' in obs_cols:
    X = pd.get_dummies(X, columns=['income']) 
    X.drop('income_-1', axis=1, inplace=True) # drugs=never is base case

# offspring
if 'offspring' in obs_cols:
    X = pd.get_dummies(X, columns=['offspring']) 
    X.drop("offspring_doesn't have kids", axis=1, inplace=True) # drugs=never is base case

# orientation
if 'orientation' in obs_cols:
    X = pd.get_dummies(X, columns=['orientation']) 
    X.drop("orientation_straight", axis=1, inplace=True) # drugs=never is base case

# pets
if 'pets' in obs_cols:
    X = pd.get_dummies(X, columns=['pets']) 
    X.drop("pets_dislikes dogs and dislikes cats", axis=1, inplace=True) # drugs=never is base case

# smokes
if 'smokes' in obs_cols:
    X = pd.get_dummies(X, columns=['smokes']) 
    X.drop("smokes_no", axis=1, inplace=True) # drugs=never is base case

# status
if 'status' in obs_cols:
    X = pd.get_dummies(X, columns=['status']) 
    X.drop("status_available", axis=1, inplace=True) # drugs=never is base case

##################################################################
# Predicted variable
prd_cols = [
    "sex",
] 
y = df[prd_cols]
##################################################################


##################################################################
# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)
##################################################################


##################################################################
# Build & Train the models
# Instantiate the model (using the default parameters)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=16)

# Fit the model, that is, train it
logreg.fit(X_train, y_train)
##################################################################


##################################################################
# Test the models:
y_pred = logreg.predict(X_test)

# Evaluation using Confusion Matrix
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

##################################################################

##################################################################
# Confusion Matrix Evaluation metrics:
from sklearn.metrics import classification_report
target_names = ['female', 'male']
print(classification_report(y_test, y_pred, target_names=target_names))

##################################################################
