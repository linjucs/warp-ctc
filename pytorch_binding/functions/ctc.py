from __future__ import division

import torch
from torch.autograd import Function
from _ext import ctc

class CTC(Function):

    def __init__(self, blank_label=None):
        """
        Constructor for CTC cost.

        Arguments:
            blank_label (optional) (Int): Integer representing the index
                of the blank, defaults to `alphabet_size - 1`.
        """
        super(CTC, self).__init__()
        self.blank_label = blank_label

    def forward(self, activations, labels, lengths, label_lengths):
        """
        Computes the CTC cost for a minibatch of examples.

        Arguments:
            activations (FloatTensor): The activations should
                un-normalized and of shape (minibatch, time, output).
            labels (IntTensor): 1D tensor of labels for each example
                consecutively.
            lengths (IntTensor): 1D tensor of number actviation time-steps
                for each example.
            label_lengths (IntTensor): 1D tensor of label lengths for
                each example.
            blank_label (optional) (Int): Integer representing the index
                of the blank, defaults to `alphabet_size - 1`.

        Returns:
            costs (FloatTensor): .
        """
        costs = torch.zeros(activations.size()[0])

        # Transpose minibatch and time for compatability with warp-ctc
        activations = torch.transpose(activations, 0, 1).contiguous()
        grads = torch.zeros(activations.size())

        blank_label = self.blank_label
        if blank_label is None:
            blank_label = activations.size()[-1] - 1
        ctc.ctc_cost_and_grad(activations, labels,
                              lengths, label_lengths,
                              costs, grads, blank_label)
        self._grads = grads
        return costs

    def backward(self, cost):
        grads = self._grads.transpose_(0, 1).contiguous()
        return grads, None, None, None

class CTCLoss(CTC):
    def __init__(self, size_average=True, blank_label=None):
        super(CTCLoss, self).__init__(blank_label)
        self.size_average = size_average

    def forward(self, *args):
        parent = super(CTCLoss, self)
        costs = parent.forward(*args)
        cost = torch.sum(costs)
        if self.size_average:
            cost = cost / costs.size(0)
        return costs.new((cost,))

    def backward(self, *args):
        parent = super(CTCLoss, self)
        grads = parent.backward(*args)[0]
        if self.size_average:
            grads = grads / grads.size()[0]
        return grads, None, None, None

