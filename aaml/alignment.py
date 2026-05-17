"""
Needleman-Wunsch global sequence alignment algorithm.

Used to compute alignment loss for training the distance metric.
"""


def needleman_wunsch(seq1, seq2, metric, gap_penalty=-1.0):
    """
    Needleman-Wunsch global alignment using a distance metric.

    Unlike traditional NW with similarity scores, this version uses
    distance: smaller distances → better match, gaps penalize.

    Args:
        seq1: first amino acid sequence (string)
        seq2: second amino acid sequence (string)
        metric: distance metric object (callable metric(a,b) -> distance)
        gap_penalty: penalty for gaps (negative for distance framework)

    Returns:
        (aligned_seq1, aligned_seq2): aligned sequences with gaps
    """
    n, m = len(seq1), len(seq2)

    # Initialize score matrix
    # Since we're dealing with distances, we want to minimize total distance
    # We use negative gap penalty (the cost of a gap)
    score = [[0.0] * (m + 1) for _ in range(n + 1)]

    # Initialize first row and column with gap penalties
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + gap_penalty
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + gap_penalty

    # Fill the score matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score[i - 1][j - 1] + _score_match(seq1[i - 1], seq2[j - 1], metric)
            gap_i = score[i - 1][j] + gap_penalty
            gap_j = score[i][j - 1] + gap_penalty
            score[i][j] = max(match, gap_i, gap_j)

    # Traceback
    aligned1, aligned2 = [], []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i][j] == score[i - 1][j - 1] + _score_match(seq1[i - 1], seq2[j - 1], metric):
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and score[i][j] == score[i - 1][j] + gap_penalty:
            aligned1.append(seq1[i - 1])
            aligned2.append('-')
            i -= 1
        else:
            aligned1.append('-')
            aligned2.append(seq2[j - 1])
            j -= 1

    # Reverse since we built from end
    return ''.join(reversed(aligned1)), ''.join(reversed(aligned2))


def _score_match(a, b, metric):
    """
    Score for aligning two amino acids.

    Since metric(a,b) is a distance (0 = identical, positive = different),
    we want to maximize score by using negative distance.
    """
    import torch
    dist = metric(a, b)
    if isinstance(dist, torch.Tensor):
        dist = dist.item()
    return -dist  # negative distance so smaller distance = higher score


def calculate_alignment_identity(seq1_aligned, seq2_aligned):
    """Calculate sequence identity from alignment."""
    matches = sum(1 for a, b in zip(seq1_aligned, seq2_aligned) if a == b and a != '-')
    length = len(seq1_aligned)
    return matches / length if length > 0 else 0.0