import torch
import torch.nn as nn
from einops import rearrange


class RearrangeModule(nn.Module):
    def __init__(self, pattern, **shapes):
        super().__init__()
        self.pattern = pattern
        self.shapes = shapes

    def forward(self, x):
        return rearrange(x, self.pattern, **self.shapes)
