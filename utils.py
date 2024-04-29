import numpy as np


def softmax(x):
    x = x - np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def normalize(x):
    max_val = np.max(x, axis=0)
    min_val = np.min(x, axis=0)
    return (x - min_val) / (max_val - min_val)


def top_k_sampling(logits, k):
    #TODO
    pass


def top_p_sampling(logits, p):
    #TODO
    pass


def temperature_scaling(logits, temperature):
    scaled_logits = logits / temperature
    scaled_probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
    return scaled_probs
