#!/usr/bin/env python


from nltk import word_tokenize
from pprint import pprint
import argparse
import nltk
import string
import pandas as pd

"""
Running:
$ python nltk-analytics.py [path_to_file]
"""

# Globals: stopwords, punctuation

punctuation = list(string.punctuation)
punctuation += ['...', "''", '``']  # To add items 
""" ['!', '#', '"', '%', '$', "'", '&', ')', '(', '+', '*', '-', ',', '/', '.', ';', ':', '=', '<', '?', 
'>', '@', '[', ']', '\\', '_', '^', '`', '{', '}', '|', '~'] """
stopwords = nltk.corpus.stopwords.words('english')
stopwords += []  # To add items 
""" [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', 
u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', 
u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', 
u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', 
u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', 
u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', 
u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', 
u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', 
u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', 
u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', 
u'don', u'should', u'now', u'd', u'll', u'm', u'o', u're', u've', u'y', u'ain', u'aren', u'couldn', u'didn', 
u'doesn', u'hadn', u'hasn', u'haven', u'isn', u'ma', u'mightn', u'mustn', u'needn', u'shan', u'shouldn', u'wasn', 
u'weren', u'won', u'wouldn'] """


def load_dataframe(filepath):
    """ Load CSV data as a pandas DataFrame 
    And apply data pre-processing and cleaning """
    if 'youtube' in filepath:
        data = pd.read_csv(filepath, encoding='utf-8')
        comments = data['top_level_comment']
        # YT comments from the API need some additional cleaning
        # Most of the text will be clean of HTML since textFormat=plainText was passed in the API call.
        comments = [unicode(c).replace(u'\xef\xbb\xbf', u'') for c in comments if c != ' - ']
        comments = [c.replace(u'\ufeff', u'') for c in comments]
        return comments
    elif 'reddit' in filepath:
        data = pd.read_csv(filepath, encoding='utf-8')
        comments = data['comment']
        return comments


def freq_dist(data):
    """ Frequency distribution """ 
    # data is a list of unicode strings: [str, str, str...]
    comments = [word_tokenize(c) for c in data]
    comments = [[w.lower() for w in c if w.lower() not in stopwords and w not in punctuation] for c in comments]
    # comments = [[w for w in c if w not in punctuation] for c in comments]

    # merged_comments = "\n".join([" ".join([w for w in c]) for c in comments])
    merged_comments = []  # Init ignore_var or else list comprehension spams the terminal output
    ignore_var = [[merged_comments.append(w) for w in c] for c in comments]

    text_FD = nltk.FreqDist(merged_comments)

    pprint(text_FD.most_common(20))


def collocations(data):
    """ N-grams """


def main(input_data):
    """ Top-level commands """
    data = load_dataframe(input_data)
    freq_dist(data)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data', help='Path to the input data file')
    # parser.add_argument('input_data_YT', help='Path to the input data file')
    # parser.add_argument('repo', help='User\'s repo')
    args = parser.parse_args()
    main(args.input_data)
    # main()
