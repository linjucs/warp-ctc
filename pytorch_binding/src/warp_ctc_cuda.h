
/*
 * The activations should be shape (time, minibatch, alphabet)
 * in C order.
 * 
 * *NB* We assume the blank_label is the last index in the alphabet.
 *
 * *NB* th_labels, th_lengths and th_label_lengths and 
 *  th_costs *must* all be in CPU memory. th_grads *must*
 *  be in same memory space as *th_activations*.
 */
void ctc_cost_and_grad_cuda(THCudaTensor *th_activations,
                       THIntTensor *th_labels,
                       THIntTensor *th_lengths,
                       THIntTensor *th_label_lengths,
                       THFloatTensor *th_costs,
                       THCudaTensor *th_grads,
                       int blank_label);
