"""
Data loading utilities for sequence alignment datasets.

Supports:
- MSF (Multiple Sequence Format) files
- FASTA files
- Generating synthetic sequence pairs for testing
"""

import random


# Small built-in test dataset with curated sequence pairs
_BUILTIN_SEQUENCES = [
    # Highly conserved sequences (close homologs)
    ("ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"),  # identical
    ("ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"),  # identity

    # Similar (conservative substitutions)
    ("MKLLVLGASRVGKSS", "MKIIVIGASRVGKSS"),  # K->I, L->I, R->G (Ras family)
    ("GIVEGCCTSICS", "GI-VGCCKSICK"),

    # Medium conservation
    ("PEPTIDEKING", "PEPTIDAKING"),
    ("ALIGNMENTTEST", "ALIGNMINTTEST"),

    # Diverse (distant homologs)
    ("MVLSEGEWQLVLHVW", "MVLSEGEWQLVLHGW"),
    ("AKVEQPMEKASR", "ATVEQPMGRASR"),

    # Random generated with some conservation
    ("ALGERIA", "ALGVBIA"),
    ("MATRIX", "METRIX"),

    # Pairs with gaps
    ("SHORTSEQ", "LONGERSEQUENCE"),
    ("ABCDEF", "ABCDGEF"),

    # Kinase-like conserved regions
    ("HRDLKPENLL", "HRDLKPENLL"),
    ("DFGLSK", "DFGLSK"),

    # Variable regions
    ("KVLADV", "KIVADV"),
    ("YAMYAS", "YALYAS"),
]


def load_builtin_data():
    """
    Load built-in test sequence pairs.

    Returns:
        list of (seq1, seq2) tuples
    """
    return _BUILTIN_SEQUENCES.copy()


def load_fasta(filename):
    """
    Load sequences from a FASTA file.

    Args:
        filename: path to FASTA file

    Returns:
        list of (header, sequence) tuples
    """
    sequences = []
    current_header = None
    current_seq = []

    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_header is not None:
                        sequences.append((current_header, ''.join(current_seq)))
                    current_header = line[1:].strip()
                    current_seq = []
                else:
                    current_seq.append(line.upper().replace(' ', ''))

        if current_header is not None:
            sequences.append((current_header, ''.join(current_seq)))
    except FileNotFoundError:
        print(f"Warning: File {filename} not found, using built-in data")
        return load_builtin_data()

    return sequences


def generate_synthetic_data(n_pairs=20, seq_length=30, mutation_rate=0.3):
    """
    Generate synthetic sequence pairs for testing.

    Args:
        n_pairs: number of sequence pairs to generate
        seq_length: length of each sequence
        mutation_rate: probability of mutation at each position

    Returns:
        list of (seq1, seq2) tuples
    """
    aa_codes = "ACDEFGHIKLMNPQRSTVWY"
    pairs = []

    for _ in range(n_pairs):
        # Generate random sequence
        seq1 = ''.join(random.choice(aa_codes) for _ in range(seq_length))

        # Mutate with given rate
        seq2_chars = []
        for char in seq1:
            if random.random() < mutation_rate:
                # Random substitution
                new_char = random.choice([c for c in aa_codes if c != char])
                seq2_chars.append(new_char)
            else:
                seq2_chars.append(char)
        seq2 = ''.join(seq2_chars)

        pairs.append((seq1, seq2))

    return pairs


def create_alignment_dataset(sequences_a, sequences_b):
    """
    Create a paired alignment dataset from two lists of sequences.

    Args:
        sequences_a: list of first sequences (or FASTA entries)
        sequences_b: list of second sequences (or FASTA entries)

    Returns:
        list of (seq1, seq2) tuples
    """
    # Handle FASTA entries: (header, seq) tuples
    if sequences_a and isinstance(sequences_a[0], tuple):
        sequences_a = [s[1] for s in sequences_a]
    if sequences_b and isinstance(sequences_b[0], tuple):
        sequences_b = [s[1] for s in sequences_b]

    return list(zip(sequences_a, sequences_b))