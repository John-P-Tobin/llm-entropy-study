"""
entropy.py
----------
Entropy helper functions.
"""

import math


def shannon_entropy(probs, normalize=True):
    """
    Compute Shannon entropy in bits.
    """
    p = [x for x in probs if x > 0]

    if normalize:
        Z = sum(p)
        p = [x / Z for x in p]

    H = -sum(x * math.log2(x) for x in p)
    return H


def surprisal(prob):
    """
    Compute the surprisal of one outcome.
    """
    return -math.log2(prob)


def max_entropy(n):
    """
    Compute the maximum entropy over n outcomes.
    """
    return math.log2(n)


if __name__ == "__main__":
    # Example distributions from testing.
    paris_probs = [
        0.999999, 0.000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    noun_probs = [
        0.236569, 0.236569, 0.208772, 0.052786, 0.052786, 0.046583,
        0.036279, 0.032016, 0.015123, 0.010394, 0.009173, 0.006304,
        0.005564, 0.003824, 0.003824, 0.003374, 0.002978, 0.002628,
        0.002047, 0.001806,
    ]

    H_paris = shannon_entropy(paris_probs)
    H_noun = shannon_entropy(noun_probs)

    print("Constrained prompt ('capital of France'):")
    print(f"    H = {H_paris:.4f} bits   (near 0 = ordered / predictable)\n")

    print("Open prompt ('any noun'):")
    print(f"    H = {H_noun:.4f} bits   (large = disordered / unpredictable)")
    print(f"    For comparison, a uniform distribution over the {len(noun_probs)} "
          f"observed tokens would have H = {max_entropy(len(noun_probs)):.4f} bits.")

