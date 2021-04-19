import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F
class custom_BN(torch.nn.Module):
    def __init__(self,num_features):
        self.num_features = num_features 
        self.alpha = nn.Parameter(torch.ones(num_features))
        self.register_parameter('alpha',self.alpha)
        self.beta = nn.Parameter(torch.ones(num_features))
        self.register_parameter('beta',self.beta)
    def forward(slef,x):
        if len(x.shape) == 4:
            return x * self.alpha.unsqueeze(1).unsqueeze(1) + self.beta.unsqueeze(1).unsqueeze(1)
        elif len(x.shape) == 3:
            return x * self.alpha.unsqueeze(1) + self.beta.unsqueeze(1)

class custom_BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                       .format(input.dim()))
    def forward(self,input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
              exponential_average_factor = self.momentum
        if self.training:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
          return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
