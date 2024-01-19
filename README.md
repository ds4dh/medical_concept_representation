# Automated phenotyping

## Project description
Repository for the manuscript entitled "Comparative neural word embeddings approaches for medical concept representation and patient trajectory prediction".
- This project aims to compare NLP models (word2vec, fastTex and GloVe), based on the quality of their representation of medical concepts.
- We use MIMIC-IV to train the models, from which we extract patient trajectories as sequences of (amongst others) ICD10 and ATC codes.
- We train the models with model-specific NLP tasks that use patient trajectory sequences as input.
- We evaluate the models by producing medical concept embeddings, clustering them, comparing them to existing biomedical terminologies, and using them for medical outcome and patient trajectory prediction tasks.

## How to reproduce the results
