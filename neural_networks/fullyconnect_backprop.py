import numpy as np

def fullyconnect_backprop(in_sensitivity, in_, weight):
    '''
    The backpropagation process of fullyconnect
      input parameter:
          in_sensitivity  : the sensitivity from the upper layer w.r.t y, shape:
                          : [number of images, number of outputs in feedforward]
          in_             : the input in feedforward process, shape: 
                          : [number of images, number of inputs in feedforward]
          weight          : the weight matrix of this layer, shape: 
                          : [number of inputs in feedforward, number of outputs in feedforward]

      output parameter:
          weight_grad     : the gradient of the weights, shape: 
                          : [number of inputs in feedforward, number of outputs in feedforward]
          bias_grad       : the gradient of the bias, shape: 
                          : [number of outputs in feedforward, 1]
          out_sensitivity : the sensitivity to the lower layer w.r.t x, shape:
                          : [number of images, number of inputs in feedforward]

    Note : remember to divide by number of images in the calculation of gradients.
    '''

    # TODO

    # begin answer
    N = in_.shape[0]
    n_in = in_.shape[1]
    n_out = in_sensitivity.shape[1]
    weight_grad = in_.T @ in_sensitivity / N
    bias_grad = np.average(in_sensitivity, axis=0).reshape((n_out, 1))
    out_sensitivity = in_sensitivity @ weight.T
    # end answer

    return weight_grad, bias_grad, out_sensitivity

