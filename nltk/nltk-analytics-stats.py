#!/usr/bin/env python


# from __future__ import division
from scipy import stats
import argparse
import numpy as np
import pandas as pd

"""
Running:
$ python nltk-analytics-stats.py [path_to_file]
"""

# Globals:



def ttest(data1, data2):
    """ """
    global d1, d2
    d1 = pd.read_csv(data1)
    d2 = pd.read_csv(data2)
    #
    d1_1 = d1['num_char']
    d2_1 = d2['num_char']
    d1_2 = d1['num_word']
    d2_2 = d2['num_word']
    d1_3 = d1['num_sent']
    d2_3 = d2['num_sent']
    d1_4 = d1['TTR']
    d2_4 = d2['TTR']
    d1_5 = d1['FRE']
    d2_5 = d2['FRE']

    print data1 + " & " + data2
    print "num_char: ", stats.ttest_ind(d1_1, d2_1, equal_var=False)
    print "num_word: ", stats.ttest_ind(d1_2, d2_2, equal_var=False)
    print "num_sent: ", stats.ttest_ind(d1_3, d2_3, equal_var=False)
    print "TTR: ", stats.ttest_ind(d1_4, d2_4, equal_var=False)
    print "FRE: ", stats.ttest_ind(d1_5, d2_5, equal_var=False)
    print " "


def main():
    """ Top-level commands """
    # print stats.ttest_ind([1, 2, 3], [2, 4, 6], equal_var=False)  # Testing

    ttest('nltk/data/desc-stats-2016-12-04-19h-43m-youtube-comments-855Am6ovK7s.csv', 
          'nltk/data/desc-stats-2016-12-04-21h-11m-reddit-comments-54nrcs.csv')

    ttest('nltk/data/desc-stats-2016-12-04-19h-43m-youtube-comments-FRlI2SQ0Ueg.csv', 
          'nltk/data/desc-stats-2016-12-04-21h-11m-reddit-comments-56psaa.csv')
    
    ttest('nltk/data/desc-stats-2016-12-04-19h-43m-youtube-comments-smkyorC5qwc.csv', 
          'nltk/data/desc-stats-2016-12-04-21h-11m-reddit-comments-58eh18.csv')

    ttest('nltk/data/desc-stats-2016-12-04-19h-43m-youtube-comments.csv', 
          'nltk/data/desc-stats-2016-12-04-21h-11m-reddit-comments.csv')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('input_data', help='Path to the input data file')
    # parser.add_argument('input_data_YT', help='Path to the input data file')
    # parser.add_argument('repo', help='User\'s repo')
    # args = parser.parse_args()
    # main(args.input_data)
    main()


"""
$ python nltk/nltk-analytics-stats.py 
nltk/data/desc-stats-2016-12-04-19h-43m-youtube-comments-855Am6ovK7s.csv & nltk/data/desc-stats-2016-12-04-21h-11m-reddit-comments-54nrcs.csv
num_char:  (1.2247772667245502, 0.2208237898693905)
num_word:  (1.1398700711666698, 0.25449797177855088)
num_sent:  (1.8697836044055591, 0.061679728825147509)
TTR:  (-2.7313092596732087, 0.0063619589068144907)
FRE:  (1.7848022645594006, 0.074438522609403424)
 
nltk/data/desc-stats-2016-12-04-19h-43m-youtube-comments-FRlI2SQ0Ueg.csv & nltk/data/desc-stats-2016-12-04-21h-11m-reddit-comments-56psaa.csv
num_char:  (-0.72897783693215168, 0.46610175908488816)
num_word:  (-0.97010693793150982, 0.33211277700991004)
num_sent:  (-1.8224902115810513, 0.068510867170723799)
TTR:  (0.26858844043184743, 0.78827076210563962)
FRE:  (1.8904391138329097, 0.058825406694714567)
 
nltk/data/desc-stats-2016-12-04-19h-43m-youtube-comments-smkyorC5qwc.csv & nltk/data/desc-stats-2016-12-04-21h-11m-reddit-comments-58eh18.csv
num_char:  (0.51919592929930658, 0.60368161567549983)
num_word:  (0.23870359706180225, 0.81135927869421243)
num_sent:  (0.62740629822235749, 0.53046181634165024)
TTR:  (-1.9453626328456757, 0.051862548381530635)
FRE:  (1.8074289902737357, 0.070837000764045158)
 
nltk/data/desc-stats-2016-12-04-19h-43m-youtube-comments.csv & nltk/data/desc-stats-2016-12-04-21h-11m-reddit-comments.csv
num_char:  (0.58615644200179351, 0.55779321784989055)
num_word:  (0.20629706481498117, 0.83656602862493779)
num_sent:  (0.42593487024049659, 0.67016993892413046)
TTR:  (-2.5690440870147357, 0.010220074819899954)
FRE:  (3.168753810575879, 0.0015379616911579508)
"""