"""
AAML: Amino Acid Metric Learning
ML library for learning distance metrics on the 20 standard amino acids.
"""

from .features import FeatureSpace
from .metric import AA_Metric
from .alignment import needleman_wunsch
from .loss import MetricLoss
from .train import train_model
from .evaluation import compare_with_blosum, matrix_to_dataframe

__version__ = "0.1.0"