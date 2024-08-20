"""
Author: Alex R. Mead
Date: August 2024

Description:
Further exploring classification with tabular and an unstructured
data related to the OkCupid Profiles dataset. 

This is a "learn burn" file which will be reformated for the 
Substack post.

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