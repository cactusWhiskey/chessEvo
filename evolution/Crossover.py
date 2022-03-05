import random


def cxUniformShallow(ind1, ind2, layer_prob):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped according to the
    *indpb* probability.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param layer_prob: Probability of crossover happening on a given layer
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    num_layers = min(len(ind1), len(ind2))

    for i in range(num_layers):  # iterate through list of layers
        if random.random() < layer_prob:  # select this layer

            # Note: ind[i] is a list of ndarrays
            ind1[i], ind2[i] = ind2[i], ind1[i]  # just swap lists of layer matrices wholesale

    return ind1, ind2


# def cxUniformDeep(ind1, ind2, indpb, layer_prob):
#     """Executes a uniform crossover that modify in place the two
#     :term:`sequence` individuals. The attributes are swapped according to the
#     *indpb* probability.
#     :param ind1: The first individual participating in the crossover.
#     :param ind2: The second individual participating in the crossover.
#     :param indpb: Independent probability for each attribute to be exchanged within a layer.
#     :param layer_prob: Probability of crossover happening on a given layer
#     :returns: A tuple of two individuals.
#     This function uses the :func:`~random.random` function from the python base
#     :mod:`random` module.
#     """
#     numLayers = min(len(ind1), len(ind2))
#
#     for i in range(numLayers):  # iterate through list of layer weights
#         if random.random() < layer_prob:  # select this layer
#             # Note: ind[i] is a list of ndarrays
#
#             matricesInLayer = min(len(ind1[i]), len(ind2[i]))
#             for j in range(matricesInLayer):
#                 # Note ind[i][j] is a single ndarray
#
#                 r,c = ind1[i][j].shape
#                 for row in r:
#                     for col in c:
#                         ind1[i][j][row,col] , ind2[i][j][row,col] = ind2[i][j][row,col] , ind1[i][j][row,col] =
#
#
#
#     return ind1, ind2
