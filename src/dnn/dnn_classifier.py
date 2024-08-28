"""
Author: Alex R. Mead
Date: August 2024

Description:
Stepping beyond the tabular data within the OkCupid dataset,
we now explore the unstructured text which is given by each
user as free-response question and answers. 

Here we will explore embeddings and a Deep Neural Network
to attemp to better classify the gender of a given user
than either the Logistic Regression or TabNet model.

"""


# The general pattern of building and training a Deep Neural Network (DNN)
# happesn in 5 steps, each performed below in turn. 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1: Preprocesing of data, partially already done in src/dnn/preprocess.py script
df = pd.read_csv("data/okcupid_profiles.csv")
essay0_idx = df[df['essay0'].notna()]

X = np.loadtxt('data/essay0_embed.csv', delimiter=",") # Load embeddings
X = X[essay0_idx.index] # Extract the non-blank rows
df['gender'] = df['sex'].apply(lambda sex: 0 if sex == 'f' else 1)
y = df[df['essay0'].notna()]['gender'].to_numpy() #Extract the labels: 'sex'

# Scale the data: I think this is already done.
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Convert to Pytorch datatypes
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.uint8)

# Make dataset with Pytorch machinery
dataset = TensorDataset(X_tensor, y_tensor)

# Split the dataset into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2: Select Classifier Model Structure
# Define the neural network. TODO: Move this to src/dnn/model.py
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)  # Adjust the output size based on the number of classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Initialize the neural network
model = SimpleNN()
