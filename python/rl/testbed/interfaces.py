#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from abc import ABC


class Randomized(ABC):
    """
    A base class for all classifiers, rankers or modules that use RNGs.

    Attributes
        random_state   The seed to be used by RNGs
    """

    random_state: int = 1
