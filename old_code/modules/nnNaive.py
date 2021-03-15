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

from six import iteritems, itervalues, string_types
from six.moves import range

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

import numpy as np
from numpy import exp, dot, zeros, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt,\
    empty, sum as np_sum, ones, logaddexp, log, outer

from scipy.special import expit

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

import time
import random
import unidecode

from . import multiSigKernel as multiSigKer
from . import sigKernel as sigKer

logger = logging.getLogger(__name__)


    ### uncomment following example code for use of nn.Sequential module

# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 6, kernel_size=5),
#             nn.ReLU(True),
#             nn.Conv2d(6, 16, kernel_size=5),
#             nn.ReLU(True))
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(16, 6, kernel_size=5),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(6, 3, kernel_size=5),
#             nn.ReLU(True))
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


class Autoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Autoencoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        # self.linear0 = nn.Linear(embedding_size*4,embedding_size*2, bias = False)
        # self.linear1 = nn.Linear(embedding_size*2,embedding_size, bias = False)
        # self.linear2 = nn.Linear(embedding_size, embedding_size*2, bias = False)
        # self.linear3 = nn.Linear(embedding_size*2, embedding_size*4, bias=False)
        self.linear4 = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, inputs):#, vocab, old_weights, old_mmd):
        out = self.embeddings(inputs)
        # out = F.relu(out)   ### I don't know if this is necessary...
        # out = self.linear0(out)
        # # out = F.relu(out)
        # out = self.linear1(out)
        # # out = F.relu(out)
        # out = self.linear2(out)
        # # out = F.relu(out)
        # out = self.linear3(out)
        # out = F.relu(out)
        out = self.linear4(out)
        out = F.softmax(out,dim=1)
        # out = F.relu(out)
        return out


class EarlyStopping():
    def __init__(self, patience=5, min_percent_gain=0.1):
        self.patience = patience
        self.loss_list = []
        self.min_percent_gain = min_percent_gain / 100.

    def update_loss(self, loss):
        self.loss_list.append(loss)
        if len(self.loss_list) > self.patience:
            del self.loss_list[0]

    def stop_training(self):
        if len(self.loss_list) == 1:
            return False
        gain = (max(self.loss_list) - min(self.loss_list)) / max(self.loss_list)
        print("Loss gain: {}%".format(round(100 * gain, 2)))
        if gain < self.min_percent_gain:
            return True
        else:
            return False


def runModel(vocab, model, epochs):
        ### we can probably get the embedding_dimension from the Word2VecKeyedVectors class (vocab.wv.size)
        ### should understand if using this object is more efficient for calculations
        losses = []
        # loss_function = nn.CrossEntropyLoss()    # implement my own loss function --> target should be a tensor
        optimizer = optim.Adam(model.parameters(), lr=0.0005)      # Adam vs SGD
        in_tensor = torch.tensor([i for i in np.arange(0,vocab.size)], dtype=torch.long)
        # early_stopping = EarlyStopping(patience=10, min_percent_gain=1)
        old_weights, old_mmd = None, None
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(epochs):
            total_loss = 0
            model.zero_grad()
            optimizer.zero_grad()

            out = model(in_tensor)#, vocab, old_weights, old_mmd)

            # loss = loss_function(out, in_tensor)#, dtype=torch.long)
            loss = MMDLoss_fn(out, in_tensor, vocab, old_weights, old_mmd)

            loss.backward()

            old_weights = torch.zeros(out.shape, dtype = torch.float)
            old_weights.data = out.clone()
            old_mmd = torch.zeros(loss.shape, dtype = torch.float)
            old_mmd.data = loss.clone()

            optimizer.step()

            total_loss += loss.item()
            losses.append(total_loss)
            # display the epoch training loss
            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, total_loss))
            # early_stopping.update_loss(np.mean(losses))
            # if early_stopping.stop_training():
            #     break
        wv = model.embeddings.weight #.mm(model.linear0.weight).mm(model.linear1,weight)
        vocab.wv.vectors = wv.detach().numpy()
        return losses

class MMDLoss(autograd.Function):
    @staticmethod
    def forward(ctx, input, target, vocab, old_weights = None, old_mmd = None):
        new_mmd = mmd(input, target, vocab)
        ctx.save_for_backward(input, new_mmd, old_weights, old_mmd)
        return new_mmd.sum()

    @staticmethod
    def backward(ctx, grad_output):
        input, new_mmd, old_weights, old_mmd = ctx.saved_tensors
        if old_weights == None or old_mmd == None:
            return torch.rand(input.size()), None, None, None, None
        else:
            grad_input = new_mmd.sub(old_mmd.div(input.sub(old_weights)))
            # grad_input = (new_mmd.sub(old_mmd)).mean().div(input.sub(old_weights))
            return grad_input, None, None, None, None

def MMDLoss_fn(input, target, vocab, old_weights, old_mmd):
    return MMDLoss.apply(input, target, vocab, old_weights, old_mmd)

def mmd(input, target, vocab):
    loss = expectedSignatures(input, target, vocab).computeExpectedSignaturesBis()
    return loss

class expectedSignatures():
    def __init__(self, input, target, vocab, max_no_pairs = 300):
        self.vocab_size = vocab.size
        self.wv = vocab.wv
        self.paths = vocab.paths
        self.out_weights = input.data
        self.max_no_pairs = max_no_pairs
        self.mmd = torch.tensor([0 for i in range(len(target))], dtype=torch.float)        # mmd distance initialized

    def computeExpectedSignatures(self):
        for index, word in enumerate(self.wv.index2word):
            kernel01 = 0    # average kernel over paths (X,X') in one-hot space
            kernel00 = 0    # average kernel over pair of paths (X,Y) with X in one-hot space, Y in transformed space
            kernel11 = 0    # average kernel over paths (Y,Y') in transformed space
            paths_no = self.wv.vocab[word].sentences_no
            d = len(paths_no)
            if not d == 1:
                if d*(d-1)/2 > self.max_no_pairs:
                    k = int(sqrt(2*self.max_no_pairs))
                    paths_to_consider = random.sample(paths_no, k)
                else:
                    k = d
                    paths_to_consider = paths_no
                count = 0
                for i in range(k):
                    for j in range(k):
                        if i < j:
                            kernel00 += sigKer.sig_kernel(self.createPath(paths_to_consider[i], one_hot = True),
                                                   self.createPath(paths_to_consider[j], one_hot = True))
                            kernel11 += sigKer.sig_kernel(self.createPath(paths_to_consider[i]),
                                                   self.createPath(paths_to_consider[j]))
                            count +=1
                assert count == k*(k-1)/2
                kernel00 = kernel00/count
                kernel11 = kernel11/count
                count = 0
                for i in range(k):
                    for j in range(k):
                        kernel01 += sigKer.sig_kernel(self.createPath(paths_to_consider[i], one_hot = True),
                                            self.createPath(paths_to_consider[j]))
                        count += 1
                assert count == k**2
                kernel01 = kernel01/count
                self.mmd[index] = kernel00 - 2 * kernel01 + kernel11
            else:
                self.mmd[index] = 0
        return self.mmd

    def createPath(self, pathIndex, one_hot = False):
        '''Outputs a NxD numpy array, where N is the length of the path (i.e. the number of points)
        and D is the dimension. Each row in the array is a point in the path. '''
        path = self.paths[pathIndex]
        pathArray = np.zeros((len(path),self.vocab_size))
        if one_hot ==True:
            dummy_indices = [i for i in range(len(path))]
            pathArray[dummy_indices,path] = 1
        else:
            for i in range(len(path)):
                pathArray[i,:] = np.array(self.out_weights[i])
        return pathArray

    def computeExpectedSignaturesBis(self):
        for index, word in enumerate(self.wv.index2word):
            paths_no = self.wv.vocab[word].sentences_no
            d = len(paths_no)
            if not d == 1:
                if d*(d-1)/2 > self.max_no_pairs:
                    k = int(sqrt(2*self.max_no_pairs))
                    paths_to_consider = random.sample(paths_no, k)
                else:
                    k = d
                    paths_to_consider = paths_no
                X,Y = self.createPaths(paths_to_consider)
                count = k*(k-1)/2
                kernel00 = multiSigKer.multi_sig_kernel(X,X,0)
                kernel00 = np.triu(kernel00-np.diag(kernel00.diagonal())).sum()/count
                kernel11 = multiSigKer.multi_sig_kernel(Y,Y,0)
                kernel11 = np.triu(kernel11-np.diag(kernel11.diagonal())).sum()/count
                kernel01 = multiSigKer.multi_sig_kernel(X,Y,0)
                kernel01 = np.triu(kernel01).mean()
                self.mmd[index] = kernel00 - 2*kernel01 + kernel11
            else:
                self.mmd[index] = 0
        return self.mmd

    def createPaths(self, pathIndices):
        '''Outputs a NxD numpy array, where N is the length of the path (i.e. the number of points)
        and D is the dimension. Each row in the array is a point in the path. '''
        pathsArrayOneHot = []
        pathsArrayOut = []
        maxPathLen = 0
        for pathIndex in pathIndices:
            path = self.paths[pathIndex]
            maxPathLen = max(len(path),maxPathLen)
            pathOneHot = np.zeros((maxPathLen,self.vocab_size))
            pathOut = np.zeros((maxPathLen,self.vocab_size))
            dummy_indices = [i for i in range(len(path))]
            pathOneHot[dummy_indices,path] = 1
            for i in range(len(path)):
                pathOut[i,:] = np.array(self.out_weights[path[i]])
            pathsArrayOneHot.append(pathOneHot)
            pathsArrayOut.append(pathOut)
        for i in range(len(pathsArrayOneHot)):
            if not pathsArrayOneHot[i].shape == (maxPathLen, self.vocab_size):
                padding = np.zeros((maxPathLen - pathsArrayOneHot[i].shape[0], self.vocab_size))
                pathsArrayOneHot[i] = np.vstack((pathsArrayOneHot[i], padding))
                pathsArrayOut[i] = np.vstack((pathsArrayOut[i], padding))
        return np.array(pathsArrayOneHot), np.array(pathsArrayOut)