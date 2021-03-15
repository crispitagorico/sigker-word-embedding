from __future__ import division  # py3 "true division"

import logging
import sys
import os
import heapq
from timeit import default_timer
from copy import deepcopy
from collections import defaultdict
import threading
import itertools
import warnings

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

import numpy as np
from numpy import exp, dot, zeros, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt,\
    empty, sum as np_sum, ones, logaddexp, log, outer

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import time

def tsne_plot(vocab, embedding_dim):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for i,word in enumerate(vocab.wv.vocab):
        tokens.append(vocab.wv.vectors[i])
        labels.append(word)
    if not embedding_dim == 2:
        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        tokens = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in tokens:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()