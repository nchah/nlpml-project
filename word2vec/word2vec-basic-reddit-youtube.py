#!/usr/bin/env python

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import math
import os
import random

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import pandas as pd
import tensorflow as tf
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
import string

stopwords = nltk.corpus.stopwords.words('english')
punctuation = set(string.punctuation)

# Step 1: Load the data.

# Reddit
# comments = pd.read_csv('reddit/data/output/2016-12-04-21h-11m-reddit-comments-54nrcs.csv', encoding='utf-8')
# comments = pd.read_csv('reddit/data/output/2016-12-04-21h-11m-reddit-comments-56psaa.csv', encoding='utf-8')
# comments = pd.read_csv('reddit/data/output/2016-12-04-21h-11m-reddit-comments-58eh18.csv', encoding='utf-8')
comments = pd.read_csv('reddit/data/output/2016-12-04-21h-11m-reddit-comments.csv', encoding='utf-8')
comments = comments['comment'].tolist()

# Youtube
# comments = pd.read_csv('youtube/data/output/2016-12-04-19h-43m-youtube-comments-855Am6ovK7s.csv', encoding='utf-8')
# comments = pd.read_csv('youtube/data/output/2016-12-04-19h-43m-youtube-comments-FRlI2SQ0Ueg.csv', encoding='utf-8')
# comments = pd.read_csv('youtube/data/output/2016-12-04-19h-43m-youtube-comments-smkyorC5qwc.csv', encoding='utf-8')
# comments = pd.read_csv('youtube/data/output/2016-12-04-19h-43m-youtube-comments.csv', encoding='utf-8')
# comments = comments['top_level_comment'].tolist()
# comments = [unicode(c).replace(u'\xef\xbb\xbf', u'') for c in comments if c != ' - ']
# comments = [c.replace(u'\ufeff', u'') for c in comments]

# Stemming 
# stemmer = LancasterStemmer()
# stems = [[stemmer.stem(word) for word in comm] for comm in comments]

# Lemmatizing the text
comments = [word_tokenize(c) for c in comments]

lemmatizer = WordNetLemmatizer()
pos_text = [nltk.pos_tag(comm) for comm in comments]
lemmas = [[lemmatizer.lemmatize(word) for word, tag in comm] for comm in pos_text]

# Merging all individual comments into a single text blob
data = '\n'.join(" ".join(comm) for comm in lemmas)

# data = ""
# for c in comments:
#     data += c + " "

# Stopwords and punctuation
punctuation.remove("'")
# for p in punctuation:
#     data = data.replace(p, '')
#     print('replaced: ' + p)

words = tf.compat.as_str(data).split()

print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 1000  # 50000


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                           num_sampled, vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.initialize_all_variables()

# Step 5: Begin training.
num_steps = 200001  # 100001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    lose_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


# Step 6: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(12, 12))  # in inches
    texts = []
    for i, label in enumerate(labels):
        pos = nltk.pos_tag([label])
        # ignore_tags = ['DT', 'PRP', 'VB', 'RB', 'IN', 'JJ']
        # if label.lower() not in stopwords and pos[0][1] not in ignore_tags and pos[0][1] == 'NN':
        if label.lower() not in stopwords \
        and label not in punctuation \
        and label[0].isupper() \
        or label.lower() in emolex_positive or label.lower() in emolex_negative:
            x, y = low_dim_embs[i, :]
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y)
    # Implements adjusted text labels from external lib. Else, activate plt.annotate() below.
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
            # plt.annotate(label,
            #              xy=(x, y),
            #              xytext=(5, 2),
            #              textcoords='offset points',
            #              ha='right',
            #              va='bottom')
    plt.savefig(filename, dpi=600)

    subprocess.call(["say 'program completed'"], shell=True)  # notification for OS X


# Using the EmoLex lexicon
emolex = pd.read_table('emolex.txt',
                        delimiter='\t',header=None, names=['word','emotion','value'],
                        dtype={'word':unicode,'emotion':unicode,'value':int},
                        encoding='utf-8')
# From the EmoLex lexicon, creating the Positive/Negative lists
emolex_positive = emolex[(emolex.emotion=='positive') & (emolex.value==1)].word.tolist()
emolex_negative = emolex[(emolex.emotion=='negative') & (emolex.value==1)].word.tolist()


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import nltk
from nltk import pos_tag
from adjustText import adjust_text
import subprocess

stopwords = nltk.corpus.stopwords.words('english')

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 600  # 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])

labels = [reverse_dictionary[i] for i in xrange(plot_only)]
labels = [unicode(word, 'utf-8') for word in labels]
print(str(labels))

print(len(low_dim_embs), len(labels))

plot_with_labels(low_dim_embs, labels)

