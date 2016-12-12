#!/usr/bin/env python


from __future__ import division
from hyphen import Hyphenator, dict_info
from nltk import word_tokenize, sent_tokenize
from nltk.collocations import *
from numpy import sum, mean
from pprint import pprint
from scipy import stats
import argparse
import csv
import nltk
import string
import numpy as np
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
stopwords += ["'s", "n't", "'m"]  # To add items 
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

# Setting up CSVs
fd = '' #'nltk/'
freq_dist_csv = fd + 'nltk-freq-dist.csv'
ngrams_csv = fd + 'nltk-ngrams.csv'
desc_stats_csv = fd + 'nltk-desc-stats.csv'

freq_dist_headers = ['id', 'word', 'frequency']
ngrams_headers = ['id', 'ngram', 'measure_value', 'measure_type', 'samples']
desc_stats_headers = ['id', 'type', 'value']

# Write headers for each script call
hr = [freq_dist_headers, ngrams_headers, desc_stats_headers]
for i, file in enumerate([freq_dist_csv, ngrams_csv, desc_stats_csv]):
    with open(file, 'a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(hr[i])


def load_dataframe(filepath):
    """ Load CSV data as a pandas DataFrame 
    And apply data pre-processing and cleaning """
    global resource_id  # For CSV normalization
    resource_id = filepath.split('/')[len(filepath.split('/'))-1] # Using input's file name

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

    merged_comments = []  # Init ignore_var or else list comprehension spams the terminal output
    ignore_var = [[merged_comments.append(w) for w in c] for c in comments]

    text_FD = nltk.FreqDist(merged_comments).most_common(20)
    # pprint(text_FD)

    for i1, i2 in text_FD:
        with open(freq_dist_csv, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([resource_id, i1, i2])


def collocations(data):
    """ N-grams """
    data2 = [c.lower() for c in data]
    comments = [word_tokenize(c) for c in data]
    comments = [[w.lower() for w in c if w not in punctuation] for c in comments]

    merged_comments = []  # Init ignore_var or else list comprehension spams the terminal output
    ignore_var = [[merged_comments.append(w) for w in c] for c in comments]

    # pprint(nltk.Text(merged_comments).collocations())

    # Simple implementation:
    # ngrams = nltk.ngrams(merged_comments, 4)
    # for ng in ngrams:
    #     pprint(ng)

    # More detailed implementation: http://www.nltk.org/_modules/nltk/metrics/association.html
    # Possible measures: chi_sq, pmi, likelihood_ratio, jaccard
    bigram_stats = nltk.collocations.BigramAssocMeasures()
    bigram_finder = BigramCollocationFinder.from_words(merged_comments)
    bigram_finder.apply_freq_filter(5)
    bigrams = bigram_finder.nbest(bigram_stats.pmi, 10)
    # for n1, n2 in bigrams:
    #     print n1, n2, bigram_finder.score_ngram(bigram_stats.pmi, n1, n2)

    for n1, n2 in bigrams:
        sample = ''
        for comm in data2:
            if n1 + " " + n2 in comm:
                sample += '- - ' + comm + '\n\n'
        score = bigram_finder.score_ngram(bigram_stats.pmi, n1, n2)
        with open(ngrams_csv, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([resource_id, n1+' '+n2, score, 'pmi', sample.encode('utf-8')])
    
    # Trigrams
    trigram_stats = nltk.collocations.TrigramAssocMeasures()
    trigram_finder = TrigramCollocationFinder.from_words(merged_comments, window_size=3)
    trigram_finder.apply_freq_filter(5)
    trigrams = trigram_finder.nbest(trigram_stats.pmi, 10)
    # for n1, n2, n3 in trigrams:
    #     print n1, n2, n3, trigram_finder.score_ngram(trigram_stats.pmi, n1, n2, n3)

    for n1, n2, n3 in trigrams:
        sample = ''
        for comm in data2:
            if n1 + " " + n2 + " " + n3 in comm:
                sample += '- - ' + comm + '\n\n'
        score = trigram_finder.score_ngram(trigram_stats.pmi, n1, n2, n3)
        with open(ngrams_csv, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([resource_id, n1+' '+n2+' '+n3, score, 'pmi', sample.encode('utf-8')])


def desc_stats(data):
    """ Descriptive statistics """
    global lens, num_words, num_sents

    lens = []
    for comm in data:
        lens.append(len(comm))
    lens = pd.DataFrame(lens)
    lens_stats = lens[0].describe().to_dict()
    # avg_ttr = pd.DataFrame(ttr(data)).mean()[0]
    # desc['avg_TTR:'] = str(avg_ttr)
    # print desc

    for i in lens_stats:
        with open(desc_stats_csv, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([resource_id, 'char_len_'+i, lens_stats[i]])

    # Stats for # of words, # of sents
    num_words = []
    for comm in data:
        num_words.append(len(word_tokenize(comm)))
    num_words = pd.DataFrame(num_words)
    num_words_stats = num_words[0].describe().to_dict()

    for i in num_words_stats:
        with open(desc_stats_csv, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([resource_id, 'word_len_'+i, num_words_stats[i]])

    num_sents = []
    for comm in data:
        num_sents.append(len(sent_tokenize(comm)))
    num_sents = pd.DataFrame(num_sents)
    num_sents_stats = num_sents[0].describe().to_dict()

    for i in num_sents_stats:
        with open(desc_stats_csv, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([resource_id, 'sent_len_'+i, num_sents_stats[i]])


def ttr(data):
    """ Type-Token Ratio """
    global ttr

    ttr = []
    for comm in data:
        text = word_tokenize(comm)
        types = set(text)
        ttr_temp = len(types)/len(text)
        ttr.append(ttr_temp)
    ttr = pd.DataFrame(ttr)
    ttr_stats = ttr[0].describe().to_dict()

    for i in ttr_stats:
        with open(desc_stats_csv, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([resource_id, 'TTR_'+i, ttr_stats[i]])

    # TTR stats calcualated for chunks of 100
    ttr_list = []
    temp = []
    for i in xrange(0, len(data), 100):
        chunk = data[i:i+100]
        types = len(set(chunk))
        tokens = len(chunk)
        ttr = types / tokens
        temp.append(ttr)
    print len(temp)
    ttr_list.append(float(np.mean(temp)))

    for i in ttr_list:
        with open(desc_stats_csv, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([resource_id, 'TTR_p100_', i])


def flesch_reading_score(data):
    """ Flesch Reading Ease Score """
    global fre

    comments = [word_tokenize(c.lower()) for c in data]
    fre = []
    # h_en = Hyphenator('en_US')  # Tried PyHyphen
    # len(h_en.syllables(u'and'))  # but not good performance on these cases

    from nltk.corpus import cmudict
    d = cmudict.dict()

    for i, comm in enumerate(comments):
        twords = len(comm)
        tsents = len(sent_tokenize(data[i]))
        tsyllb_temp = []
        for w in comm:
            if w in d:
                phonemes = d[w]
                syllables = [sum(x[-1].isdigit() for x in p) for p in phonemes]
                tsyllb_temp.append(max(syllables))
        tsyllb = np.sum(tsyllb_temp)

        FRE = 206.835 - (1.015 * (twords / tsents)) - (84.6 * (tsyllb / twords))
        fre.append(FRE)
    fre = pd.DataFrame(fre)
    fre_stats = fre[0].describe().to_dict()

    for i in fre_stats:
        with open(desc_stats_csv, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([resource_id, 'FRE_'+i, fre_stats[i]])


def output_dataframe(data):
    """ Output and save the DataFrame as CSV. Useful to see distribution of each row's value """
    global lens, num_words, num_sents, ttr, fre

    comments = data
    final_pd = pd.DataFrame(comments)
    final_pd['num_char'] = lens
    final_pd['num_word'] = num_words
    final_pd['num_sent'] = num_sents
    final_pd['TTR'] = ttr
    final_pd['FRE'] = fre

    final_pd.to_csv('desc-stats-' + resource_id, encoding='utf-8')


def main(input_data):
    """ Top-level commands """
    comments_data = load_dataframe(input_data)
    freq_dist(comments_data)
    collocations(comments_data)
    desc_stats(comments_data)
    ttr(comments_data)
    flesch_reading_score(comments_data)
    output_dataframe(comments_data)

    print("Done: " + input_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data', help='Path to the input data file')
    # parser.add_argument('input_data_YT', help='Path to the input data file')
    # parser.add_argument('repo', help='User\'s repo')
    args = parser.parse_args()
    main(args.input_data)
    # main()
