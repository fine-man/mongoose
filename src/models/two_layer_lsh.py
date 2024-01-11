import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lsh_layer import LSHLayer

class TwoLayerLSH(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_classes,
                 K,
                 L,
                 hash_weight=None,
                 flatten_first=False,
                 threads=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.K = K
        self.L = L
        self.hash_weight = hash_weight
        self.flatten = flatten_first

        self.hidden_fc =  LSHLayer(input_size, hidden_size, K, L,
                                   hash_weight, threads)

        self.output_fc = nn.Linear(hidden_size, num_classes)

        # Variables to log extra information
        self.train_iters = 0

    def rebuild_lsh(self):
        self.hidden_fc.rebuild()

    def forward(self, X):
        if self.flatten:
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