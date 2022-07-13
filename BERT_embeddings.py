# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:24:23 2022

@author: caberg
"""

samp = moddat.sample(n=10)

samp['email_content_BERT']

from sentence_transformers import SentenceTransformer
sentences = [samp['email_content_BERT'].iloc[0]]
sentences

sentences

model = SentenceTransformer('C:/Users/caberg/Downloads/all-mpnet-base-v2 (1)')
embeddings = model.encode(sentences)

samp['embeddings'] = samp['email_content_BERT'].apply(lambda x: model.encode(x))
samp['embeddings']

print(embeddings)












from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('C:/Users/caberg/Downloads/all-mpnet-base-v2 (1)')
model = AutoModel.from_pretrained('C:/Users/caberg/Downloads/all-mpnet-base-v2 (1)')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)




