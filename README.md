# Automated phenotyping

## Description

This project aims to build powerful representations of patient data.
We use MIMIC-IV to train our models, from which we extract patient admissions as sequences of (amongst others) ICD and ATC codes.
We train the models with unsupervised (model-specific) NLP tasks that uses the patient admission sequences as input.
We test the models by producing medical concept embeddings, clustering them, and comparing to existing phenotypes.
The NLP models we use are GloVe, FastText, ELMO and BERT. Model trainings are categorized in two pairs of categories:
- Static vs. contextualized embeddings ({GloVe, FastText} vs. {ELMO, BERT})
- Code vs. sub-code level tokenization (here we need to see how to apply this to all models without altering their nature)
- We also propose a tokenization strategy that is specific to the hierarchy of ICD and ATC codes.

## To do

### Datasets

#### (1) Flexible data pipeline (Dimitris + Alban)
Build a pipeline that can generate a dataset with any feature. The feature choice is a vector of boolean values.

#### (2) Genetic algorithm for feature selection (Anthony)
Build a genetic algorithm that sends a boolean vector request to the dataset pipeline, and then performs diagnose prediction with tf-idf. The goal is to identify the set of features that yield the best performance for the other models, as well as to set a baseline performance for model comparison.

### Models

#### (3) Test metric (Alban)
For each model, code a test metric function that extracts embeddings from the test set (using input without diagnoses) and performs diagnose prediction using the minimal cosine distance.


## Instructions 
### setup 
> conda env create -f environment.yml 