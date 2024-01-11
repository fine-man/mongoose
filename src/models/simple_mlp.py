import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_classes,
                 flatten_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.flatten_first = flatten_first

        self.hidden_fc = nn.Linear(input_size, hidden_size)

        self.output_fc = nn.Linear(hidden_size, num_classes)

        # Variables to log extra information
        self.train_iters = 0

    def rebuild_lsh(self):
        self.hidden_fc.rebuild()

    def forward(self, X):
        if self.flatten_first:
            X = X.view(-1, 28 * 28)
            N, D = X.shape[:2]

        if self.training:
            self.train_iters += 1

        # logits.shape = (N, num_neurons)
        out = self.hidden_fc(X)

        out = F.relu(out) # (N, num_neurons)
        out = self.output_fc(out) # (N, num_classes)
        return out