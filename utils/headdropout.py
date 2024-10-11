import torch
import torch.nn as nn

class HeadDropout(nn.Module):
    def __init__(self, p=0.5):
        super(HeadDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def forward(self, x):
        # If in evaluation mode, return the input as-is
        if not self.training:
            return x
        
        # Create a binary mask of the same shape as x
        binary_mask = (torch.rand_like(x) > self.p).float()
        
        # Set dropped values to negative infinity during training
        return x * binary_mask + (1 - binary_mask) * -1e20

