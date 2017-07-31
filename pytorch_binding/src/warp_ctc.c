#include <TH/TH.h>

#include "ctc.h"

void ctc_cost_and_grad(THFloatTensor *th_activations,
                       THIntTensor *th_labels,
                       THIntTensor *th_lengths,
                       THIntTensor *th_label_lengths,
                       THFloatTensor *th_costs,
                       THFloatTensor *th_grads,
                       int blank_label)
{
    // TODO, should we move this into python??
    THArgCheck(th_activations->nDimension == 3, 0,
               "activations must have 3 dimensions.");
    THArgCheck(th_labels->nDimension == 1, 1,
               "labels must have 1 dimension.");
    THArgCheck(th_lengths->nDimension == 1, 2,
               "lengths must have 1 dimension.");
    THAssertMsg(THIntTensor_isSameSizeAs(th_lengths, th_label_lengths),
               "label_lengths must have the same size as lengths.");
    THArgCheck(th_costs->nDimension == 1, 4,
               "costs must have 1 dimension.");
    THAssertMsg(THFloatTensor_isSameSizeAs(th_activations, th_grads),
                "grads must have the same size as activations.");
    THArgCheck(THFloatTensor_isContiguous(th_activations), 0,
               "activations must be contiguous.");
    THArgCheck(THIntTensor_isContiguous(th_labels), 1,
               "labels must be contiguous.");
    THArgCheck(THIntTensor_isContiguous(th_lengths), 2,
               "lengths must be contiguous.");
    THArgCheck(THIntTensor_isContiguous(th_label_lengths), 3,
               "label_lengths must be contiguous.");
    THArgCheck(THFloatTensor_isContiguous(th_costs), 4,
               "costs must be contiguous.");
    THArgCheck(THFloatTensor_isContiguous(th_grads), 5,
               "grads must be contiguous.");


    int num_examples = THFloatTensor_size(th_activations, 1);
    int alphabet_size = THFloatTensor_size(th_activations, 2);

    ctcOptions options;
    options.loc = CTC_CPU;
    options.num_threads = 1;
    options.blank_label = blank_label;


    size_t cpu_alloc_bytes;

    float *activations = THFloatTensor_data(th_activations);
    int *lengths = THIntTensor_data(th_lengths);
    int *labels = THIntTensor_data(th_labels);
    int *label_lengths = THIntTensor_data(th_label_lengths);

    get_workspace_size(label_lengths, lengths,
                       alphabet_size, num_examples, options,
                       &cpu_alloc_bytes);

    void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);

    float *costs = THFloatTensor_data(th_costs);
    float *grads = THFloatTensor_data(th_grads);

    compute_ctc_loss(activations, grads,
                     labels, label_lengths,
                     lengths,
                     alphabet_size,
                     num_examples,
                     costs,
                     ctc_cpu_workspace,
                     options);
    free(ctc_cpu_workspace);
}
