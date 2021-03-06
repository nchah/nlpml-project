# nlpml-project

A Natural Language Processing and Machine Learning project done in part for a University of Toronto graduate course, POL2578: Computational Text Analysis.

# Dependencies

This project makes use of a number of libraries and tools.

In the Python environment, the following are used:

General Category | Library
--- | ---
Data Generation | `praw`, `requests`
Data Processing | `pandas`, `numpy`
Natural Langauge Processing | `nltk`
Machine Learning | `sklearn`, `tensorflow`


# Pipeline

This project uses the following pipeline to generate, process, and analyze the data.

## Data Generation

First, the textual data is obtained using the Reddit and YouTube APIs.
The scripts in this repository are written to first get the comments data from these platforms.

## Machine Learning

Next, the machine learning techniques are applied using the specialized libraries: NLTK (Natural Language Toolkit), Sklearn (Scikit Learn), and TensorFlow (Python API) among others.
The machine learning methods used for this project include supervised methods (classification, SVM) and unsupervised methods (PCA, clustering, word2vec).

# CLI Scripts

The following command-line interface (CLI) scripts were used for this project according to the following use cases.

Script | Use Case
--- | ---
`knowledge-graph/knowledge-graph-api.py` | Gets named entities using the Google Knowledge Graph API.
`nltk/nltk-analytics.py` | Gets various analytics measures on the text data.
`nltk/nltk-scripts.sh` | Bash script to run a set of the nltk-analytics commands.
`reddit/reddit-api.py` | Gets Reddit data using the Reddit API (with the PRAW library).
`sklearn/sklearn-cluster-*.py` | (Unsupervised ML) Implements various clustering and visualization operations. Implements Principal Components Analysis (PCA).
`sklearn/sklearn-lda-*.py` | (Unsupervised ML) Latent Dirichlet Allocation. WIP and not implemented for the project.
`sklearn/sklearn-svm.py` | (Supervised ML) Implements a Support Vector Machine to distinguish the origin of comments. Treats comment origin (Reddit or YouTube) as the label.
`word2vec/word2vec-basic-*.py` | (Unsupvervised ML) TensorFlow word2vec implementation with modifications for this project.
`youtube/youtube-data-api.py` | (Unsupervised ML) Gets YouTube data using the YouTube Data API.

# Findings





# References

I found the following awesome-* GitHub READMEs to be useful in the early exploratory stages:

- [caesar0301/awesome-public-datasets](https://github.com/caesar0301/awesome-public-datasets)
- [keonkim/awesome-nlp](https://github.com/keonkim/awesome-nlp)

