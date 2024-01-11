import os
import sys

CUR_DIR = os.getcwd()
sys.path.append(os.path.join(CUR_DIR, "lsh_lib"))
sys.path.append(os.path.join(CUR_DIR, "mongoose_slide"))

from mongoose_slide.slide_lib.lsh import LSH
from mongoose_slide.slide_lib.simHash import SimHash
from mongoose_slide.slide_lib.projectionHash import RandomProjection

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class LSHLayer(nn.Module):
    def __init__(self, in_features, out_features, K, L, hash_weight=None, threads_=4):
        super().__init__()
        self.D = in_features
        self.K = K
        self.L = L
        self.out_features = out_features
        self.hash_weight = hash_weight
        self.threads_ = threads_

        self.store_query = True
        # last layer
        self.params = nn.Linear(in_features, out_features) # weight.shape = (C, D)
        self.params.bias = nn.Parameter(torch.Tensor(out_features, 1)) # (C, 1)
        self.init_weights(self.params.weight, self.params.bias)

        # construct lsh using triplet weight
        self.lsh = None
        self.initializeLSH()

        self.count = 0
        self.sample_size = 0

        # Some legacy thresh stuff
        # TODO: checkout this thresh thing later
        # self.thresh_hash = SimHash(self.D+1, 1, self.L)
        self.thresh_hash = RandomProjection(self.D+1, 1, self.L)

        self.thresh = 0.3
        self.hashcodes = self.thresh_hash.hash(torch.cat((self.params.weight, self.params.bias), dim = 1))

        # Variables to log extra information
        # stores the activation frequency of each neuron of hidden layer
        self.activ_freq = np.zeros(out_features)
        self.train_iters = 0
        # total number of neurons activated across all training forward passes
        self.total_activ = 0
        self.avg_activ = 0 # average activations per forward pass in training
        self.last_neurons = [] # set of neuron activations in the last forward pass

        # Things to log
        # how many times each neuron is getting activated throughout training
        # total number of training iterations
        # what is the average number of neuron activations per forward pass
        # Number of times each neuron is changing it's hashcodes
        # Number of times a neuron is activated for a particular class

    def initializeLSH(self):
        self.lsh = LSH( RandomProjection(self.D+1, self.K, self.L), self.K, self.L,
                       threads_=self.threads_)
        weight_tolsh = torch.cat( (self.params.weight, self.params.bias), dim = 1)
        self.lsh.insert_multi(weight_tolsh.to(device).data, self.out_features )

    def init_weights(self, weight, bias):
        initrange = 1.0
        torch.nn.init.uniform_(weight.data, -1, 1)
        # weight.data = torch.randn(weight.shape)
        bias.data.fill_(0)
        # bias.require_grad0ent = False
 
    def rebuild(self):
        weight_tolsh = torch.cat((self.params.weight, self.params.bias), dim=1)
        self.lsh.clear()
        self.lsh.insert_multi(weight_tolsh.to(device).data, self.out_features )

    def train_forward(self, x):
        N, D = x.size()

        # query_lolash.shape = (N, D + 1)
        query_tolsh = torch.cat( (x, torch.ones(N).unsqueeze(dim = 1).to(device)), dim = 1 )

        # sid is a set containing all the neuron IDs
        # hashcode is of shape (N, L) containing the hash-buckets for each datapoint
        sid, hashcode = self.lsh.query_multi(query_tolsh.data, N)
        sid_list = list(sid)

        sample_ids = Variable(torch.from_numpy(np.asarray(sid_list, dtype=np.int64)), requires_grad=False).to(device)
        sample_size = sample_ids.size(0)

        if self.training:
            self.activ_freq[sample_ids.cpu().numpy()] += 1
            self.train_iters += 1
            self.total_activ += len(sample_ids)
            self.avg_activ = self.total_activ/self.train_iters
        self.last_neurons = sample_ids.cpu().numpy()

        sample_bias = self.params.bias.squeeze()[sample_ids]
        sample_weights = F.embedding(sample_ids, self.params.weight, sparse=True) # (num_neurons, D)

        # (N, num_neurons) = (N, D) @ (D, num_neurons)
        sample_product = x @ sample_weights.T
        sample_logits = sample_product + sample_bias

        return sample_logits, sample_ids

    def forward(self, x):
        return self.train_forward(x)

class MNISTNET(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_classes,
                 K,
                 L,
                 hash_weight=None,
                 threads=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.hash_weight = hash_weight

        self.hidden_fc =  LSHLayer(input_size, hidden_size, K, L,
                                   hash_weight, threads)

        self.output_fc = nn.Linear(hidden_size, num_classes)

        # Variables to log extra information
        self.train_iters = 0

    def rebuild_lsh(self):
        self.hidden_fc.rebuild()

    def forward(self, X):
        X = X.view(-1, 28 * 28)
        N, D = X.shape[:2]

        # sample_logits.shape = (N, num_neurons)
        sample_logits, sample_ids = self.hidden_fc(X)

        if self.training:
            self.train_iters += 1

        sample_logits = F.relu(sample_logits)

        # weights.shape = (num_classes, num_neurons)
        weights = self.output_fc.weight[:, sample_ids]
        bias = self.output_fc.bias[:]

        out = sample_logits @ weights.T
        out = out + bias
        return out