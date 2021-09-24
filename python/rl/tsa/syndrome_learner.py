#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""

from sklearn.base import ClassifierMixin

from rl.tsa.cython.model import RuleListBuilder
from rl.tsa.cython.rule_evaluation_label_wise import RegularizedLabelWiseRuleEvaluationFactory
from rl.tsa.cython.statistics_label_wise import LabelWiseStatisticsProviderFactory
from rl.common.cython.head_refinement import NoHeadRefinementFactory, FullHeadRefinementFactory
from rl.common.cython.model import ModelBuilder
from rl.common.cython.rule_induction import TopDownRuleInduction, SequentialRuleModelInduction
from rl.common.cython.sampling import NoInstanceSubSamplingFactory
from rl.common.cython.sampling import NoPartitionSamplingFactory
from rl.common.cython.thresholds_exact import ExactThresholdsFactory
from rl.common.rule_learners import FEATURE_SUB_SAMPLING_RANDOM
from rl.common.rule_learners import MLRuleLearner, SparsePolicy
from rl.common.rule_learners import create_feature_sub_sampling_factory, create_max_conditions, \
    create_stopping_criteria, create_min_support, get_preferred_num_threads


class SyndromeLearner(MLRuleLearner, ClassifierMixin):
    """
    A scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
    classification rules.
    """

    def __init__(self, from_year: int, from_week: int, to_year: int, to_week: int, random_state: int = 1,
                 feature_format: str = SparsePolicy.AUTO.value, max_rules: int = 1000, time_limit: int = -1,
                 feature_sub_sampling: str = FEATURE_SUB_SAMPLING_RANDOM, min_support: float = 0.0,
                 max_conditions: int = -1, num_threads_refinement: int = 1):
        """
        :param max_rules:                           The maximum number of rules to be induced (including the default
                                                    rule)
        :param time_limit:                          The duration in seconds after which the induction of rules should be
                                                    canceled
        :param feature_sub_sampling:                The strategy that is used for sub-sampling the features each time a
                                                    classification rule is refined. Must be `random-feature-selection`
                                                    or None, if no sub-sampling should be used. Additional arguments may
                                                    be provided as a dictionary, e.g.
                                                    `random-feature-selection{\"sample_size\":0.5}`
        :param min_support:                         The minimum fraction of the training examples that must be covered
                                                    by a rule. Must be in [0, 1)
        :param max_conditions:                      The maximum number of conditions to be included in a rule's body.
                                                    Must be at least 1 or -1, if the number of conditions should not be
                                                    restricted
        :param num_threads_refinement:              The number of threads to be used to search for potential refinements
                                                    of rules or -1, if the number of cores that are available on the
                                                    machine should be used
        """
        super().__init__(random_state, feature_format)
        self.from_year = from_year
        self.from_week = from_week
        self.to_year = to_year
        self.to_week = to_week
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.feature_sub_sampling = feature_sub_sampling
        self.min_support = min_support
        self.max_conditions = max_conditions
        self.num_threads_refinement = num_threads_refinement

    def get_name(self) -> str:
        name = 'from-year=' + str(self.from_year)
        if self.from_week >= 0:
            name += '_from-week=' + str(self.from_week)
        name += '_to-year=' + str(self.to_year)
        if self.to_week >= 0:
            name += '_to-week=' + str(self.from_week)
        name += '_max-rules=' + str(self.max_rules)
        if self.feature_sub_sampling is not None:
            name += '_feature-sub-sampling=' + str(self.feature_sub_sampling)
        if int(self.min_support) < 1:
            name += '_min-support=' + str(self.min_support)
        if int(self.max_conditions) != -1:
            name += '_max-conditions=' + str(self.max_conditions)
        if int(self.random_state) != 1:
            name += '_random_state=' + str(self.random_state)
        return name

    def _create_model_builder(self) -> ModelBuilder:
        return RuleListBuilder()

    def _create_rule_model_induction(self, num_labels: int) -> SequentialRuleModelInduction:
        stopping_criteria = create_stopping_criteria(int(self.max_rules), int(self.time_limit))
        instance_sub_sampling_factory = NoInstanceSubSamplingFactory()
        feature_sub_sampling_factory = create_feature_sub_sampling_factory(self.feature_sub_sampling)
        partition_sampling_factory = NoPartitionSamplingFactory()
        default_rule_head_refinement_factory = NoHeadRefinementFactory()
        head_refinement_factory = FullHeadRefinementFactory()
        rule_evaluation_factory = RegularizedLabelWiseRuleEvaluationFactory()
        statistics_provider_factory = LabelWiseStatisticsProviderFactory(rule_evaluation_factory,
                                                                         rule_evaluation_factory)
        thresholds_factory = ExactThresholdsFactory()
        min_support = create_min_support(self.min_support)
        max_conditions = create_max_conditions(self.max_conditions)
        num_threads_refinement = get_preferred_num_threads(self.num_threads_refinement)
        rule_induction = TopDownRuleInduction(min_support, max_conditions, num_threads_refinement)
        return SequentialRuleModelInduction(statistics_provider_factory, thresholds_factory, rule_induction,
                                            default_rule_head_refinement_factory, head_refinement_factory,
                                            instance_sub_sampling_factory, feature_sub_sampling_factory,
                                            partition_sampling_factory, stopping_criteria)
