# Automated phenotyping

## Description

This is a project that aims to build powerful representations of patient data.
We use MIMIC-IV to train our models, from which we extract patient admissions as sequences of ICD / GSN codes.
We train the models with unsupervised (model-specific) NLP tasks on the patient admission sequences.
We test the models by producing medical concept embeddings, clustering them, and comparing to existing phenotypes.
The NLP models we use are GloVe, FastText, ELMO and BERT. Model trainings are categorized in two pairs of categories:
- Static vs. contextualized embeddings ({GloVe, FastText} vs. {ELMO, BERT})
- Code vs. sub-code level tokenization (here we need to see how to apply this to all models without altering their nature)

## Datasets
Datasets are in .pickle format and have been uploaded as a zip archive to Google Drive:
- Download the zip archive in /data, using the [following link](https://drive.google.com/file/d/1NFUnnOLFuPIcrBHYVgWN_UClDlf4mXzG/view?usp=sharing)
- Run `python data/data_utils.py` to unzip the datasets and create a .json version

## To do

### Code a dataset pipeline

#### Standard input format

The dataset should consist in a list of list of codes. The first list indexes admissions and the second indexes events within each admission.
We should re-use the data pipeline of Fernando and revise which features of MIMIC-IV we want to use to build our dataset.
The idea is to build a single dataloader for all models. The model specific operations are defined in their forward functions. 

#### Standard output format

Train time: models are trained in different ways, but usually models produce logits for code prediction that are used to compute the loss.
Test time: each model should provide a function that writes concept embeddings in a [Gensim](https://radimrehurek.com/gensim/) .txt file.

### Code the models

#### Individual models

Each model should be a standard PyTorch model and follow the input / output specifications defined above.

#### Training and testing

A single PyTorch Lightning (PL) wrapper (see example in train.py) will then handle the interactions between all models and the dataset.
PL defines all experiment scripts such as training (reconstruct patient admission sequences) and testing (producing concept embeddings) functions.
