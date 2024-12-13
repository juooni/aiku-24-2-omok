
"""import tensorflow as tf

def softmax_cross_entropy_with_logits(y_true, y_pred):

	p = y_pred
	pi = y_true

	zero = tf.zeros(shape = tf.shape(pi), dtype=tf.float32)
	where = tf.equal(pi, zero)

	negatives = tf.fill(tf.shape(pi), -100.0) 
	p = tf.where(where, negatives, p)

	loss = tf.nn.softmax_cross_entropy_with_logits(labels = pi, logits = p)

	return loss

"""
import torch
import torch.nn.functional as F

def softmax_cross_entropy_with_logits(y_true, y_pred):
    """
    y_true: Tensor of shape (batch_size, num_classes), containing probabilities (e.g., one-hot vectors)
    y_pred: Tensor of shape (batch_size, num_classes), raw logits output by the model
    """
    p = y_pred
    pi = y_true

    zero = torch.zeros_like(pi)
    where = pi == zero  # Boolean tensor where pi == 0

    negatives = torch.full_like(pi, -100.0)
    p = torch.where(where, negatives, p)

    # Compute log softmax of p
    log_softmax_p = F.log_softmax(p, dim=1)

    # Compute cross entropy loss
    # For each sample, compute -sum(pi * log_softmax_p, dim=1)
    loss = -torch.sum(pi * log_softmax_p, dim=1)

    # Return mean loss over the batch
    return loss.mean()
