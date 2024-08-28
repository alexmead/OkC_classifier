"""
Author: Alex R. Mead
Date: August 2024

Description:
Preprocess script to transform the unstructured free response
data into embeddings.

"""

import pandas as pd

##################################################################
# Load the dataset
df = pd.read_csv("data/okcupid_profiles.csv")

# uuid's for each row
import uuid
df['uuid'] = [uuid.uuid4() for _ in range(len(df))]
df.set_index('uuid', inplace=True)
# save df locally, only need to do this once, as you want a unique UUID
df.to_csv("data/ok_cupid_profiles_uuid.csv")
##################################################################

# Loop over each essay and get the embedding
essay_cols = ['essay'+str(num) for num in range(0,10)]
# df = df.drop(essay_cols, axis=1)

essay_cols = 'essay0'


##################################################################