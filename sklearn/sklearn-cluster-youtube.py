#!/usr/bin/env python

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.pipeline import Pipeline
# from __future__ import print_function
import argparse
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nltk
import numpy as np
import pandas as pd

""" Implements Unsupervised ML method: Clustering
Running:
$ python sklearn-cluster-youtube.py [path_to_input_file]
"""


# Stopwords - Sklearn's stopwords package
stopwords = list(text.ENGLISH_STOP_WORDS)

# Stopwords - NLTK implementation
stopwords = nltk.corpus.stopwords.words('english')


# def process_kmeans(corpus):
#     # tfidf = TfidfVectorizer(stop_words='english', max_features=200)
#     tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=0.7, min_df=10, max_features=100)
#     x = tfidf.fit_transform(corpus)
#     km = KMeans(n_clusters=3)
#     km.fit(x)
#     km.predict(x)


def plot_scores_and_clusters_from_kmeans(corpus):
    """ Plotting silhouette scores and the clusters
    Adapted from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html """
    # range_n_clusters = [2, 3]
    range_n_clusters = [2, 3, 4, 5, 6]
    for n_clusters in range_n_clusters:
        # Start Tfidf
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=0.7, min_df=10, max_features=100)
        x = tfidf.fit_transform(corpus)

        # Initialize the clusterer with n_clusters value
        km = KMeans(n_clusters=n_clusters)
        km.fit(x)
        cluster_labels = km.predict(x)

        # Saving the data with predicted classes
        newdata = pd.DataFrame({'label' : km.predict(x), 'text' : corpus})
        newdata.to_csv('yt-km-labeled-clusters-' + str(n_clusters) + '.csv', encoding='utf-8')
        
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(x, km.labels_)
        print("KM: For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        with open('yt-km-silhouette-scores-labeled-clusters.csv', 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([n_clusters, silhouette_avg])

        # Get most important words in each cluster
        centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = tfidf.get_feature_names()
        for i in range(n_clusters):
            print "Cluster %d:" % (i+1) 
            for ind in centroids[i,  :10]:
                with open('yt-km-top-terms-labeled-clusters-' + str(n_clusters) + '.csv', 'a') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow([n_clusters, i, terms[ind]])
                print ' %s' % terms[ind] 


def process_pca(corpus):
    """ Implementing PCA """
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=0.7, min_df=10, max_features=100)
    x = tfidf.fit_transform(corpus)
    pca = PCA(n_components=2)
    X = x.toarray()
    pca.fit(X)
    components = pca.fit_transform(X)

    newdata = pd.DataFrame({'z1': components[:,0], 'z2': components[:,1], 'text' : corpus})
    newdata.to_csv('yt-pca-components.csv', encoding='utf-8')

    # TODO: make into separate function
    # loadings = pca.components_.argsort()[::-1][:10]
    # terms = tfidf.get_feature_names()
    # for c in range(2):
    #     print "Component %i" % c
    #     for ind in loadings[c,:]:
    #         print " %s : %0.3f" %(terms[ind], pca.components_[c,ind])
    #         with open('yt-pca-components-and-terms.csv', 'a') as csv_file:
    #             csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #             csv_writer.writerow([c, terms[ind], pca.components_[c,ind]])

    return components


def plot_scores_and_clusters_from_pca(X, corpus):
    """ Plotting silhouette scores and the clusters
    Adapted from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html """
    # range_n_clusters = [2, 3]
    range_n_clusters = [2, 3, 4, 5, 6]
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        fig.subplots_adjust(wspace=0.3)  # Adjust width space betweens subplots
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.2, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        # Initialize the clusterer with n_clusters value
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)

        # Saving the data with predicted classes
        newdata = pd.DataFrame({'label' : cluster_labels, 'text' : corpus})
        newdata.to_csv('yt-pca-labeled-clusters-' + str(n_clusters) + '.csv', encoding='utf-8')
        
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("PCA: For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title("Silhouette plot for the various clusters", fontsize=11)
        ax1.set_xlabel("Silhouette coefficient values, Avg: " + str(round(silhouette_avg, 4)), fontsize=11)
        ax1.set_ylabel("Cluster label", fontsize=11)
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)
        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=200)
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%  d$' % i, alpha=1, s=50)
        ax2.set_title("Visualization of the clustered data", fontsize=11)
        ax2.set_xlabel("Feature space for the 1st feature", fontsize=11)
        ax2.set_ylabel("Feature space for the 2nd feature", fontsize=11)
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=12, fontweight='bold')
        plt.show(block=False)
        plt.savefig('yt-pca-clusters-' + str((n_clusters)) + '.png', dpi=600)


def load_dataframe(filepath):
    """ Load CSV data as a pandas DataFrame 
    And apply data pre-processing and cleaning """
    data = pd.read_csv(filepath, encoding='utf-8')
    comments = data['top_level_comment']
    # YT comments from the API need some additional cleaning
    # Most of the text will be clean of HTML since textFormat=plainText was passed in the API call.
    comments = [unicode(c).replace(u'\xef\xbb\xbf', u'') for c in comments if c != ' - ']
    comments = [c.replace(u'\ufeff', u'') for c in comments]
    return comments


def main(input_data):
    """ """
    data = load_dataframe(input_data)
    # data = load_dataframe('youtube/data/output/2016-12-04-19h-43m-youtube-comments-855Am6ovK7s.csv')

    data2 = process_pca(data)
    plot_scores_and_clusters_from_pca(data2, data)

    plot_scores_and_clusters_from_kmeans(data)

    return ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data', help='Path to the input data file')
    # parser.add_argument('repo', help='User\'s repo')
    args = parser.parse_args()
    main(args.input_data)
    # main()
