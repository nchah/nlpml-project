#!/usr/bin/env python

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit, train_test_split
# Using sklearn.cross_validation brings Deprecation warning 
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
import pandas as pd
import numpy as np

""" Implements Support Vector Machhine on aggregated comments data
Running:
$ python sklearn-svm.py [path_to_Reddit_data] [path_to_YouTube_data]
"""


def load_dataframe_YT(filepath):
    """ Load CSV data as a pandas DataFrame 
    And apply data pre-processing and cleaning """
    data = pd.read_csv(filepath, encoding='utf-8')
    comments = data['top_level_comment'].tolist()
    # YT comments from the API need some additional cleaning
    # Most of the text will be clean of HTML since textFormat=plainText was passed in the API call.
    comments = [unicode(c).replace(u'\xef\xbb\xbf', u'') for c in comments if c != ' - ']
    comments = [c.replace(u'\ufeff', u'') for c in comments]
    return comments


def load_dataframe_RD(filepath):
    """ Load CSV data as a pandas DataFrame
    And apply data pre-processing and cleaning """
    data = pd.read_csv(filepath, encoding='utf-8')
    comments = data['comment'].tolist()
    return comments


def merge_data(rd_data, yt_data):
    """ Merge the X, Y datasets """
    # Defining globals for wider access
    global x_rd, x_yt, all_x, all_y

    x_rd = rd_data
    y_rd = ['RD' for i in range(len(rd_data))]
    x_yt = yt_data
    y_yt = ['YT' for i in range(len(yt_data))]
    print "- Sample of RD:", x_rd[:5]
    print "- Sample of YT:", x_yt[:5]

    min_val = min([len(x_rd) ,len(x_yt)])
    # print min_val

    # Merging all the data into one list of lists. Item orders are maintained.
    all_x = []
    [[all_x.append(i) for i in l] for l in [x_rd[:min_val], x_yt[:min_val]]]
    all_y = []
    [[all_y.append(i) for i in l] for l in [y_rd[:min_val], y_yt[:min_val]]]
    if len(all_x) == len(all_y):
        print "- Merge completed:", len(all_x), "total, with each corpus:", min_val
    # all_x = [all_x, all_y]
    # z, all_y = np.unique(all_y, return_inverse=True)


def svm():
    """ Implement SVM """
    global x_train, x_test, y_train, y_test, all_x, all_y, classifier1, tfidf

    tfidf = TfidfVectorizer(stop_words='english')
    # tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=0.7, min_df=10, max_features=100)
    print "- " + str(tfidf)  # For logging the settings on command line output
    X = tfidf.fit_transform(all_x)

    # Setting training and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(X, all_y, test_size=0.3, random_state=0)
    print "- x train:", x_train.shape, '\n', \
          "- x test:", x_test.shape, '\n', \
          "- y train:", len(y_train), '\n', \
          "- y test:", len(y_test)
    # print y_train  # Verified that distribution of labels was randomized

    # Set the SVC type to use: LinearSVC or SVC
    classifier1 = LinearSVC()
    # classifier1 = SVC(C=1, kernel='rbf')
    print "- " + str(classifier1)  # For logging the settings on command line output
    classifier1.fit(x_train, y_train)

    yhat = classifier1.predict(x_test)
    # print yhat  # For debugging

    # Predict new values using trained SVM model  # TODO: new function
    # print classifier1.predict(tfidf.transform(["sounds care comedy makes lies win presidential america president "]))
    
    acc = metrics.accuracy_score(y_test, yhat)
    f1 = metrics.f1_score(y_test, yhat, average='weighted')
    print "The percent correctly predicted is %0.2f%%." %(acc*100)
    print "The F1 score is %0.3f." %(f1)

    mc = metrics.classification_report(y_test, yhat)
    print mc


def show_top_features(classifier, vectorizer, categories):
    """ Get top 10 
    Note: for LinearSVC(), as per http://scikit-learn.org/stable/modules/svm.html
    'LinearSVC implements "one-vs-the-rest" multi-class strategy, thus training n_class models. 
    If there are only two classes, only one model is trained' """
    feature_names = np.asarray(vectorizer.get_feature_names())
    # for i, category in enumerate(categories):
    for category in categories:
        # print classifier.coef_
        top10 = np.argsort(classifier.coef_[0])[-30:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))


def most_informative_feature_for_class_svm(vectorizer, classifier, n):
    """ """
    global x_rd, x_yt
    labelid = 0 # this is the coef we're interested in. 
    feature_names = vectorizer.get_feature_names()
    svm_coef = np.asarray(classifier.coef_)#.toarray() 
    topn = sorted(zip(svm_coef[labelid], feature_names))[-n:]
    print '%-15s%-12s%-12s%-12s' % ("Feature", "Coef", "RD counts", "YT counts")
    rd_count = 0
    yt_count = 0
    for coef, feat in topn:
        for comm in x_rd:
            if feat in comm.lower():
                rd_count += 1
        for comm in x_yt:
            if feat in comm.lower():
                yt_count += 1
        print '%-15s%-12f%-12d%-12d' % (feat, coef, rd_count, yt_count)
        # print feat, '\t', coef, '\t', rd_count, '\t', yt_count
        rd_count, yt_count = 0, 0


def main(input_data_RD, input_data_YT):
    """ Top-level commands """
    global classifier1, tfidf, all_y

    yt = load_dataframe_YT(input_data_YT)
    rd = load_dataframe_RD(input_data_RD)
    merge_data(rd, yt)
    svm()

    # cats = ['YT']
    # show_top_features(classifier1, tfidf, cats)
    most_informative_feature_for_class_svm(tfidf, classifier1, 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data_RD', help='Path to the input data file')
    parser.add_argument('input_data_YT', help='Path to the input data file')
    # parser.add_argument('repo', help='User\'s repo')
    args = parser.parse_args()
    main(args.input_data_RD, args.input_data_YT)
    # main()
