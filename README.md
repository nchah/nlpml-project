# nlpml-project

Natural language processing project

# Pipeline

This project uses the following pipeline to generate, process, and analyze the data.

## Data Generation

First, the textual data is obtained using the Reddit and YouTube APIs.
The scripts in this repository are written to first get the comments data from these platforms.

## Machine Learning

Next, the machine learning techniques are applied using the specialized libraries: NLTK (Natural Language Toolkit), Sklearn (Scikit Learn), and TensorFlow (Python API). 


# Scripts

Script | Use Case
--- | ---
`knowledge-graph/knowledge-graph-api.py` | Gets named entities using the Google Knowledge Graph API.
`reddit/reddit-api.py` | Gets Reddit data using the Reddit API (with the PRAW library).
`sklearn/sklearn-cluster.py` | Implements various clustering and visualization operations.
`word2vec/word2vec-basic-*.py` | TensorFlow word2vec implementation with modifications for this project.
`youtube/youtube-data-api.py` | Gets YouTube data using the YouTube Data API.



# References

I found the following awesome-* READMEs to be useful:

- [caesar0301/awesome-public-datasets](https://github.com/caesar0301/awesome-public-datasets)
- [keonkim/awesome-nlp](https://github.com/keonkim/awesome-nlp)




