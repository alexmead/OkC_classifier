"""
Author: Alex R. Mead
Date: August 2024

Description:
Loop over each essay and get the embeddings for each essay.

"""

import pandas as pd
df = pd.read_csv("data/ok_cupid_profiles_uuid.csv")

from src.dnn.utils import (
    embed_dims, 
    essay_prompts, 
    get_embedding,
)

import numpy as np
# preallocate numpy array
X = np.empty((df.shape[0], embed_dims), np.float32)

prompt = essay_prompts['essay2']

blank_rsp = get_embedding(prompt, "")

for idx, row in df.iterrows():
    rsp = row['essay2']
    # print(f"{idx}, {rsp}")
    if pd.isna(rsp):
        embedding = blank_rsp
    else: 
        embedding = get_embedding(prompt, rsp)
    X[idx] = embedding
    if idx % 100 == 0:
        print(f"{idx+1} / 59946")

np.savetxt("data/essay2_embed.csv", X, delimiter=",", fmt="%f")




