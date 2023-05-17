# Automated phenotyping
Repository for the manuscript entitled "Comparative neural word embeddings approaches for medical concept representation and patient trajectory prediction".
- This project aims to compare NLP models (word2vec, fastTex and GloVe), based on the quality of the representation of medical concepts.
- We use MIMIC-IV to train our models, from which we extract patient admissions as sequences of (amongst others) ICD10 and ATC codes.
- We train the models with model-specific NLP tasks that uses the patient admission sequences as input.
- We test the models by producing medical concept embeddings, clustering them, comparing them to existing biomedical terminologies, and using them for various prediction tasks.