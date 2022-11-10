import numpy as np

#######################################################
# put `w1_sigmoid_forward` and `w1_sigmoid_grad_input` here #
#######################################################


def w1_sigmoid_forward(x_input):
    """sigmoid nonlinearity
    # Arguments
        x_input: np.array of size `(n_objects, n_in)`
    # Output
        the output of sigmoid layer
        np.array of size `(n_objects, n_in)`
    """
    #################
    ### YOUR CODE ###
    #################
    output = 1. / (1. + np.exp(-x_input))
    return output


def w1_sigmoid_grad_input(x_input, grad_output):
    """sigmoid nonlinearity gradient. 
        Calculate the partial derivative of the loss 
        with respect to the input of the layer
    # Arguments
        x_input: np.array of size `(n_objects, n_in)`
        grad_output: np.array of size `(n_objects, n_in)` 
            dL / df
    # Output
        the partial derivative of the loss 
        with respect to the input of the function
        np.array of size `(n_objects, n_in)` 
        dL / dh
    """
    #################
    ### YOUR CODE ###
    #################
    df_dh = w1_sigmoid_forward(x_input) * (1 - w1_sigmoid_forward(x_input))
    grad_input = grad_output * df_dh
    return grad_input


#######################################################
# put `w1_nll_forward` and `w1_nll_grad_input` here    #
#######################################################


def w1_nll_forward(target_pred, target_true):
    """Compute the value of NLL
        for a given prediction and the ground truth
    # Arguments
        target_pred: predictions - np.array of size `(n_objects, 1)`
        target_true: ground truth - np.array of size `(n_objects, 1)`
    # Output
        the value of NLL for a given prediction and the ground truth
        scalar
    """
    #################
    ### YOUR CODE ###
    #################
    output = -np.sum(target_true.T @ np.log(target_pred) + (1 - target_true).T @ np.log(1 - target_pred)) / len(target_true)
    return output



def w1_nll_grad_input(target_pred, target_true):
    """Compute the partial derivative of NLL
        with respect to its input
    # Arguments
        target_pred: predictions - np.array of size `(n_objects, 1)`
        target_true: ground truth - np.array of size `(n_objects, 1)`
    # Output
        the partial derivative 
        of NLL with respect to its input
        np.array of size `(n_objects, 1)`
    """
    #################
    ### YOUR CODE ###
    #################
    grad_input = ((target_pred - target_true) / (target_pred * (1 - target_pred))) / len(target_true)
    return grad_input


