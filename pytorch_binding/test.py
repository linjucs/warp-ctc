from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.autograd as autograd

from functions.ctc import CTC, CTCLoss

def softmax(acts):
    e_acts = np.exp(acts)
    return e_acts / np.sum(e_acts, axis=1, keepdims=True)

def wrap_and_call(ctc_fn, acts, labels, lengths):
    acts = autograd.Variable(torch.FloatTensor(acts),
                             requires_grad=True)
    label_lengths = [len(l) for l in labels]
    labels = [l for label in labels for l in label]
    labels = autograd.Variable(torch.IntTensor(labels))
    lengths = autograd.Variable(torch.IntTensor(lengths))
    label_lengths = autograd.Variable(torch.IntTensor(label_lengths))

    costs = ctc_fn(acts, labels, lengths, label_lengths)

    # Compute the gradient w.r.t. the activations
    cost = torch.sum(costs)
    cost.backward()

    return costs.data.numpy(), acts.grad.data.numpy()

def small_test():
    acts = np.array([[0.1, 0.6, 0.1, 0.1, 0.1],
                     [0.1, 0.1, 0.6, 0.1, 0.1]])
    probs = softmax(acts)

    labels = [[1, 2]]
    expected_cost = -np.log(probs[0, 1] * probs[1, 2]);

    lengths = [acts.shape[0]]
    acts = acts[None, ...]

    ctc_fn = CTC(blank_label=0)
    cost, _ = wrap_and_call(ctc_fn, acts, labels, lengths)

    assert np.allclose(cost, expected_cost, rtol=1e-7), \
        "small_test costs mismatch."

def big_test():
    alphabet_size = 6;
    T = 5;
    minibatch = 2;

    # minibatch x T x alphabet_size
    activations = [
            [[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
             [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
             [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
             [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
             [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],

            [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
             [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
             [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
             [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
             [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]]

    # From warp-ctc test_cpu (originally from tensorflow)
    expected_grads = [
            [[-0.366234, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
             [0.111121, -0.411608, 0.278779, 0.0055756, 0.00569609, 0.010436],
             [0.0357786, 0.633813, -0.678582, 0.00249248, 0.00272882, 0.0037688],
             [0.0663296, -0.356151, 0.280111, 0.00283995, 0.0035545, 0.00331533],
             [-0.541765, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],

            [[-0.69824, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
             [0.24082, -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549],
             [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, -0.797544],
             [0.280884, -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345],
             [-0.576714, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]]

    labels = [[0, 1, 2, 1, 0],
              [0, 1, 1, 0]]

    l_probs = [activations[0][t][labels[0][t]] for t in range(T)]
    expected_costs = [-np.log(np.prod(l_probs)), 5.42262]

    acts = np.log(activations)
    lengths = [T] * minibatch

    ctc_fn = CTC()
    costs, grad = wrap_and_call(ctc_fn, acts, labels, lengths)
    assert np.allclose(costs, expected_costs, rtol=1e-6), \
        "big_test costs mismatch."

    assert np.allclose(grad, expected_grads), \
        "big_test grads mismatch."

    ctc_fn = CTCLoss()

    costs, grad = wrap_and_call(ctc_fn, acts, labels, lengths)
    expected_cost = sum(expected_costs) / minibatch
    expected_grad = np.array(expected_grads) / minibatch

    assert np.allclose(costs, expected_cost, rtol=1e-6), \
        "big_test average costs mismatch."

    assert np.allclose(grad, expected_grad), \
        "big_test grads for average cost mismatch."

if __name__ == "__main__":
    small_test()
    big_test()
    print("Tests passed!")
