#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
import os
from abc import abstractmethod
from ast import literal_eval
from enum import Enum
from typing import List

import numpy as np
from scipy.sparse import issparse, isspmatrix_lil, isspmatrix_coo, isspmatrix_dok, isspmatrix_csc, isspmatrix_csr
from sklearn.utils import check_array

from rl.common.arrays import enforce_dense
from rl.common.cython.input import CContiguousLabelMatrix
from rl.common.cython.input import DokNominalFeatureMask, EqualNominalFeatureMask
from rl.common.cython.input import FortranContiguousFeatureMatrix, CscFeatureMatrix
from rl.common.cython.model import ModelBuilder
from rl.common.cython.rule_induction import RuleModelInduction
from rl.common.cython.sampling import FeatureSubSamplingFactory, RandomFeatureSubsetSelectionFactory, \
    NoFeatureSubSamplingFactory
from rl.common.cython.stopping import StoppingCriterion, SizeStoppingCriterion, TimeStoppingCriterion
from rl.common.learners import Learner, NominalAttributeLearner
from rl.common.types import DTYPE_UINT32, DTYPE_FLOAT32

FEATURE_SUB_SAMPLING_RANDOM = 'random-feature-selection'

ARGUMENT_SAMPLE_SIZE = 'sample_size'


class SparsePolicy(Enum):
    AUTO = 'auto'
    FORCE_SPARSE = 'sparse'
    FORCE_DENSE = 'dense'


class SparseFormat(Enum):
    CSC = 'csc'
    CSR = 'csr'


def create_sparse_policy(policy: str) -> SparsePolicy:
    try:
        return SparsePolicy(policy)
    except ValueError:
        raise ValueError('Invalid matrix format given: \'' + str(policy) + '\'. Must be one of ' + str(
            [x.value for x in SparsePolicy]))


def create_feature_sub_sampling_factory(feature_sub_sampling: str) -> FeatureSubSamplingFactory:
    if feature_sub_sampling is None:
        return NoFeatureSubSamplingFactory()
    else:
        prefix, args = parse_prefix_and_dict(feature_sub_sampling, [FEATURE_SUB_SAMPLING_RANDOM])

        if prefix == FEATURE_SUB_SAMPLING_RANDOM:
            sample_size = get_float_argument(args, ARGUMENT_SAMPLE_SIZE, 0.0, lambda x: 0 <= x < 1)
            return RandomFeatureSubsetSelectionFactory(sample_size)
        raise ValueError('Invalid value given for parameter \'feature_sub_sampling\': ' + str(feature_sub_sampling))


def create_stopping_criteria(max_rules: int, time_limit: int) -> List[StoppingCriterion]:
    stopping_criteria: List[StoppingCriterion] = []

    if max_rules != -1:
        if max_rules > 0:
            stopping_criteria.append(SizeStoppingCriterion(max_rules))
        else:
            raise ValueError('Invalid value given for parameter \'max_rules\': ' + str(max_rules))

    if time_limit != -1:
        if time_limit > 0:
            stopping_criteria.append(TimeStoppingCriterion(time_limit))
        else:
            raise ValueError('Invalid value given for parameter \'time_limit\': ' + str(time_limit))

    return stopping_criteria


def create_min_support(min_support: float) -> float:
    if min_support < 0 or min_support >= 1:
        raise ValueError('Invalid value given for parameter \'min_support\': ' + str(min_support))

    return min_support


def create_max_conditions(max_conditions: int) -> int:
    if max_conditions != -1 and max_conditions < 1:
        raise ValueError('Invalid value given for parameter \'max_conditions\': ' + str(max_conditions))

    return max_conditions


def create_max_head_refinements(max_head_refinements: int) -> int:
    if max_head_refinements != -1 and max_head_refinements < 1:
        raise ValueError('Invalid value given for parameter \'max_head_refinements\': ' + str(max_head_refinements))

    return max_head_refinements


def get_preferred_num_threads(num_threads: int) -> int:
    if num_threads == -1:
        return os.cpu_count()
    if num_threads < 1:
        raise ValueError('Invalid number of threads given: ' + str(num_threads))

    return num_threads


def parse_prefix_and_dict(string: str, prefixes: List[str]) -> [str, dict]:
    for prefix in prefixes:
        if string.startswith(prefix):
            suffix = string[len(prefix):].strip()

            if len(suffix) > 0:
                return prefix, literal_eval(suffix)

            return prefix, {}

    return None, None


def get_string_argument(args: dict, key: str, default: str, validation=None) -> str:
    if args is not None and key in args:
        value = str(args[key])

        if validation is not None and not validation(value):
            raise ValueError('Invalid value given for string argument \'' + key + '\': ' + str(value))

        return value

    return default


def get_bool_argument(args: dict, key: str, default: bool) -> bool:
    if args is not None and key in args:
        return bool(args[key])

    return default


def get_int_argument(args: dict, key: str, default: int, validation=None) -> int:
    if args is not None and key in args:
        value = int(args[key])

        if validation is not None and not validation(value):
            raise ValueError('Invalid value given for int argument \'' + key + '\': ' + str(value))

        return value

    return default


def get_float_argument(args: dict, key: str, default: float, validation=None) -> float:
    if args is not None and key in args:
        value = float(args[key])

        if validation is not None and not validation(value):
            raise ValueError('Invalid value given for float argument \'' + key + '\': ' + str(value))

        return value

    return default


def should_enforce_sparse(m, sparse_format: SparseFormat, policy: SparsePolicy, dtype,
                          sparse_values: bool = True) -> bool:
    """
    Returns whether it is preferable to convert a given matrix into a `scipy.sparse.csr_matrix`,
    `scipy.sparse.csc_matrix` or `scipy.sparse.dok_matrix`, depending on the format of the given matrix and a given
    `SparsePolicy`:

    - If the given policy is `SparsePolicy.AUTO`, the matrix will be converted into the given sparse format, if
      possible, if the sparse matrix is expected to occupy less memory than a dense matrix. To be able to convert the
      matrix into a sparse format, it must be a `scipy.sparse.lil_matrix`, `scipy.sparse.dok_matrix` or
      `scipy.sparse.coo_matrix`. If the given sparse format is `csr` or `csc` and the matrix is a already in that
      format, it will not be converted.

    - If the given policy is `SparsePolicy.FORCE_DENSE`, the matrix will always be converted into the specified sparse
    format, if possible.

    - If the given policy is `SparsePolicy.FORCE_SPARSE`, the matrix will always be converted into a dense matrix.

    :param m:               A `np.ndarray` or `scipy.sparse.matrix` to be checked
    :param sparse_format:   The `SparseFormat` to be used
    :param policy:          The `SparsePolicy` to be used
    :param dtype            The type of the values that should be stored in the matrix
    :param sparse_values:   True, if the values must explicitly be stored when using a sparse format, False otherwise
    :return:                True, if it is preferable to convert the matrix into a sparse matrix of the given format,
                            False otherwise
    """
    if not issparse(m):
        # Given matrix is dense
        if policy != SparsePolicy.FORCE_SPARSE:
            return False
    elif (isspmatrix_csr(m) and sparse_format == SparseFormat.CSR) or (
            isspmatrix_csc(m) and sparse_format == SparseFormat.CSC):
        # Matrix is a `scipy.sparse.csr_matrix` or `scipy.sparse.csc_matrix` and is already in the given sparse format
        return policy != SparsePolicy.FORCE_DENSE
    elif isspmatrix_lil(m) or isspmatrix_coo(m) or isspmatrix_dok(m):
        # Given matrix is in a format that might be converted into the specified sparse format
        if policy == SparsePolicy.AUTO:
            num_pointers = m.shape[1 if sparse_format == SparseFormat.CSC else 0]
            size_int = np.dtype(DTYPE_UINT32).itemsize
            size_data = np.dtype(dtype).itemsize if sparse_values else 0
            num_non_zero = m.nnz
            size_sparse = (num_non_zero * size_data) + (num_non_zero * size_int) + (num_pointers * size_int)
            size_dense = np.prod(m.shape) * size_data
            return size_sparse < size_dense
        else:
            return policy == SparsePolicy.FORCE_SPARSE

    raise ValueError(
        'Matrix of type ' + type(m).__name__ + ' cannot be converted to format \'' + str(sparse_format) + '\'')


class MLRuleLearner(Learner, NominalAttributeLearner):
    """
    A scikit-multilearn implementation of a rule learning algorithm for multi-label classification or ranking.

    Attributes
        predictions_ The predictions for the training data
    """

    def __init__(self, random_state: int, feature_format: str):
        """
        :param random_state:    The seed to be used by RNGs. Must be at least 1
        :param feature_format:  The format to be used for the feature matrix. Must be 'sparse', 'dense' or 'auto'
        """
        super().__init__()
        self.random_state = random_state
        self.feature_format = feature_format

    def _fit(self, x, y):
        # Validate feature matrix and convert it to the preferred format...
        x_sparse_format = SparseFormat.CSC
        x_sparse_policy = create_sparse_policy(self.feature_format)
        x_enforce_sparse = should_enforce_sparse(x, sparse_format=x_sparse_format, policy=x_sparse_policy,
                                                 dtype=DTYPE_FLOAT32)
        x = self._validate_data((x if x_enforce_sparse else enforce_dense(x, order='F', dtype=DTYPE_FLOAT32)),
                                accept_sparse=(x_sparse_format.value if x_enforce_sparse else False),
                                dtype=DTYPE_FLOAT32, force_all_finite='allow-nan')
        num_features = x.shape[1]

        if issparse(x):
            x_data = np.ascontiguousarray(x.data, dtype=DTYPE_FLOAT32)
            x_row_indices = np.ascontiguousarray(x.indices, dtype=DTYPE_UINT32)
            x_col_indices = np.ascontiguousarray(x.indptr, dtype=DTYPE_UINT32)
            feature_matrix = CscFeatureMatrix(x.shape[0], x.shape[1], x_data, x_row_indices, x_col_indices)
        else:
            feature_matrix = FortranContiguousFeatureMatrix(x)

        # Validate label matrix and convert it to the preferred format...
        y = check_array(y.toarray(order='C'), accept_sparse=False, ensure_2d=False, dtype=DTYPE_UINT32)
        num_labels = y.shape[1]
        label_matrix = CContiguousLabelMatrix(y)

        # Create a mask that provides access to the information whether individual features are nominal or not...
        if self.nominal_attribute_indices is None or len(self.nominal_attribute_indices) == 0:
            nominal_feature_mask = EqualNominalFeatureMask(False)
        elif len(self.nominal_attribute_indices) == num_features:
            nominal_feature_mask = EqualNominalFeatureMask(True)
        else:
            nominal_feature_mask = DokNominalFeatureMask(self.nominal_attribute_indices)

        # Induce rules...
        rule_model_induction = self._create_rule_model_induction(num_labels)
        model_builder = self._create_model_builder()
        model = rule_model_induction.induce_rules(nominal_feature_mask, feature_matrix, label_matrix, self.random_state,
                                                  model_builder)
        self.predictions_ = rule_model_induction.predictions
        return model

    def _predict(self, x):
        raise NotImplementedError('Prediction for unseen data not supported!')

    @abstractmethod
    def _create_rule_model_induction(self, num_labels: int) -> RuleModelInduction:
        """
        Must be implemented by subclasses in order to create the algorithm that should be used for inducing a rule
        model.

        :param num_labels:  The number of labels in the training data set
        :return:            The algorithm for inducting a rule model that has been created
        """
        pass

    @abstractmethod
    def _create_model_builder(self) -> ModelBuilder:
        """
        Must be implemented by subclasses in order to create the builder that should be used for building the model.

        :return: The builder that has been created
        """
        pass
