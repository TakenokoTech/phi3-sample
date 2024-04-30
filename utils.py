import numpy as np

from entity.config import Config
from entity.model import Model


def softmax(x):
    x = x - np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def normalize(x):
    max_val = np.max(x, axis=0)
    min_val = np.min(x, axis=0)
    return (x - min_val) / (max_val - min_val)


def temperature_scaling(logits):
    scaled_logits = logits / Config.temperature
    scaled_probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
    return scaled_probs


def top_k_sampling(logits):
    probs = temperature_scaling(logits)
    top_k_indices = np.argsort(probs)[::-1][:Config.k]
    top_k_probs = probs[top_k_indices]
    top_k_probs /= np.sum(top_k_probs)
    print("[top-k]", [f"[{Model.decode(i)}] {p:.3f}" for (i, p) in zip(top_k_indices, top_k_probs)][:5])
    return np.random.choice(top_k_indices, p=top_k_probs)


def top_p_sampling(logits):
    probs = temperature_scaling(logits)
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    indices_to_keep = np.where(cumulative_probs <= Config.p)[0]
    top_p_probs = sorted_probs[indices_to_keep]
    top_p_probs /= np.sum(top_p_probs)
    indices_to_token = [np.argmax(np.isclose(probs, t)) for t in sorted_probs]
    print("[top-p]", [f"[{Model.decode(indices_to_token[i])}] {p:.3f}" for (i, p) in zip(indices_to_keep, top_p_probs)][:5])
    if len(indices_to_keep) == 0:
        return indices_to_token[0]
    else:
        choice = np.random.choice(indices_to_keep, p=top_p_probs)
        return indices_to_token[choice]


def apply_repetition_penalty(logits: np.array, past: [int]):
    if len(past) < 1:
        return logits
    return np.array([
        l if i not in past[-10:-1] else l / Config.repetition_penalty
        for (i, l) in enumerate(logits)
    ])
