#!/usr/bin/env python

from sklearn.feature_extraction.text import TfidfVectorizer




tfidf = TfidfVectorizer(stop_words='english', max_features=200)
x = tfidf.fit_transform(corpus)
