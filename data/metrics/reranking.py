import numpy as np


# TODO: Search (retrieval): instead of using 1 / token_count as a weight, use
#       learnt parameters that maximize, say, recall at 50
# TODO: Reranking: train an svm or a linear classifier that computes similarity
#       between input (patient) embedding and searched output embeddings
# IDEA: Train both search and reranking models jointly