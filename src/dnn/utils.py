"""
Author: Alex R. Mead
Date: August 2024

Description: 
Utilities for the Deep Neural Network (DNN) classifier.

"""

def start_eGPU():
    """Running locally, this will ping the eGPU"""



# eGPU embedding 
import json
import requests

def get_eGPU_embedding(text):
    """Hits the local eGPU hosted embedding."""
    url = 'http://192.168.1.6:11434/api/embed' # used for WiFi
    # url = 'http://192.168.1.16:11434/api/embed' # used for LAN
    payload = {
        "model": "mxbai-embed-large:latest",
        "input": text
    }
    data = json.dumps(payload)
    response = requests.post(url, data=data, headers={'Content-Type': 'application/json'})
    rsp = json.loads(response.text)
    return rsp['embeddings'][0]


def get_embedding(prompt, user_response):
    """Returns an embedding vectory based on the reponse and response, currrently uses default eGPU model.
    
    NOTE: This is custom function for usage on my local setup. You can easily replace the following code with
    OpenAI, or whatever your favorite embeddings model is. Please note, you will need to adjust embeddings
    dimensions of the DNN to meet your needs.
    """
    text = prompt + user_response
    embedding = get_eGPU_embedding(text)
    return embedding

embed_dims = 1024

essay_prompts = {
    "essay0" : "My self summary... ",
    "essay1" : "What I’m doing with my life... ",
    "essay2" : "I’m really good at... "
}

