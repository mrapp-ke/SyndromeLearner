"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from rl.common.cython._types cimport float32
from rl.common.cython.head_refinement cimport HeadRefinementFactory
from rl.common.cython.sampling cimport InstanceSubSamplingFactory, FeatureSubSamplingFactory, PartitionSamplingFactory
from rl.common.cython.statistics cimport StatisticsProviderFactory
from rl.common.cython.stopping cimport StoppingCriterion
from rl.common.cython.thresholds cimport ThresholdsFactory

from cython.operator cimport dereference, postincrement

from libcpp.memory cimport make_unique, make_shared
from libcpp.utility cimport move


cdef class RuleInduction:
    """
    A wrapper for the pure virtual C++ class `IRuleInduction`.
    """
    pass


cdef class TopDownRuleInduction(RuleInduction):
    """
    A wrapper for the C++ class `TopDownRuleInduction`.
    """

    def __cinit__(self, float32 min_support, intp max_conditions, uint32 num_threads):
        """
        :param min_support:             The minimum fraction of the training examples that must be covered by a rule.
                                        Must be at least 1
        :param max_conditions:          The maximum number of conditions to be included in a rule's body. Must be at
                                        least 1 or -1, if the number of conditions should not be restricted
        :param num_threads:             The number of CPU threads to be used to search for potential refinements of a
                                        rule in parallel. Must be at least 1
        """
        self.rule_induction_ptr = <shared_ptr[IRuleInduction]>make_shared[TopDownRuleInductionImpl](
            min_support, max_conditions, num_threads)


cdef class Predictions:

    def __cinit__(self):
        self.ground_truth = None
        self.predictions = []

    def __getstate__(self):
        return self.ground_truth, self.predictions

    def __setstate__(self, state):
        self.ground_truth = state[0]
        self.predictions = state[1]


cdef class RuleModelInduction:
    """
    A wrapper for the pure virtual C++ class `IRuleModelInduction`.
    """

    def __cinit__(self):
        self.predictions = Predictions.__new__(Predictions)

    cpdef RuleModel induce_rules(self, NominalFeatureMask nominal_feature_mask, FeatureMatrix feature_matrix,
                                 LabelMatrix label_matrix, uint32 random_state, ModelBuilder model_builder):
        cdef shared_ptr[IRuleModelInduction] rule_model_induction_ptr = self.rule_model_induction_ptr
        cdef unique_ptr[RNG] rng_ptr = make_unique[RNG](random_state)
        cdef unique_ptr[RuleModelImpl] rule_model_ptr = rule_model_induction_ptr.get().induceRules(
            nominal_feature_mask.nominal_feature_mask_ptr, feature_matrix.feature_matrix_ptr,
            label_matrix.label_matrix_ptr, dereference(rng_ptr.get()),
            dereference(model_builder.model_builder_ptr.get()),
            wrapPredictionVisitor(<void *> self, <PredictionCythonVisitor> self.__visit_ground_truth),
            wrapPredictionVisitor(<void *> self, <PredictionCythonVisitor> self.__visit_prediction))
        cdef RuleModel model = RuleModel()
        model.model_ptr = move(rule_model_ptr)
        return model

    cdef __visit_ground_truth(self, const vector[uint32]& vec):
        cdef list value_list = []

        cdef vector[uint32].const_iterator it = vec.const_begin()
        cdef vector[uint32].const_iterator end = vec.const_end()
        cdef uint32 value

        while it != end:
            value = dereference(it)
            value_list.append(value)
            postincrement(it)

        cdef Predictions predictions = self.predictions
        predictions.ground_truth = value_list

    cdef __visit_prediction(self, const vector[uint32]& vec):
        cdef list value_list = []

        cdef vector[uint32].const_iterator it = vec.const_begin()
        cdef vector[uint32].const_iterator end = vec.const_end()
        cdef uint32 value

        while it != end:
            value = dereference(it)
            value_list.append(value)
            postincrement(it)

        cdef Predictions predictions = self.predictions
        cdef list prediction_list = predictions.predictions
        prediction_list.append(value_list)


cdef class SequentialRuleModelInduction(RuleModelInduction):
    """
    A wrapper for the C++ class `SequentialRuleModelInduction`.
    """

    def __cinit__(self, StatisticsProviderFactory statistics_provider_factory, ThresholdsFactory thresholds_factory,
                  RuleInduction rule_induction, HeadRefinementFactory default_rule_head_refinement_factory,
                  HeadRefinementFactory head_refinement_factory,
                  InstanceSubSamplingFactory instance_sub_sampling_factory,
                  FeatureSubSamplingFactory feature_sub_sampling_factory,
                  PartitionSamplingFactory partition_sampling_factory, list stopping_criteria):
        """
        :param statistics_provider_factory:             A factory that allows to create a provider that provides access
                                                        to the statistics which serve as the basis for learning rules
        :param thresholds_factory:                      A factory that allows to create objects that provide access to
                                                        the thresholds that may be used by the conditions of rules
        :param rule_induction:                          The algorithm that should be used to induce rules
        :param default_rule_head_refinement_factory:    The factory that allows to create instances of the class that
                                                        implements the strategy that should be used to find the head of
                                                        the default rule
        :param head_refinement_factory:                 The factory that allows to create instances of the class that
                                                        implements the strategy that should be used to find the heads of
                                                        rules
        :param instance_sub_sampling_factory:           The factory that should be used for creating the implementation
                                                        to be used for sub-sampling the training examples each time a
                                                        new classification rule is learned
        :param feature_sub_sampling_factory:            The factory that should be used for creating the implementation
                                                        to be used for sub-sampling the features each time a
                                                        classification rule is refined
        :param partition_sampling_factory:              The factory that should be used for creating the implementation
                                                        to be used for partitioning the training examples into a
                                                        training set and a holdout set
        :param stopping_criteria                        A list that contains the stopping criteria that should be used
                                                        to decide whether additional rules should be induced or not
        """

        cdef unique_ptr[forward_list[shared_ptr[IStoppingCriterion]]] stopping_criteria_ptr = make_unique[forward_list[shared_ptr[IStoppingCriterion]]]()
        cdef uint32 num_stopping_criteria = len(stopping_criteria)
        cdef StoppingCriterion stopping_criterion
        cdef uint32 i

        for i in range(num_stopping_criteria):
            stopping_criterion = stopping_criteria[i]
            stopping_criteria_ptr.get().push_front(stopping_criterion.stopping_criterion_ptr)

        self.rule_model_induction_ptr = <shared_ptr[IRuleModelInduction]>make_shared[SequentialRuleModelInductionImpl](
            statistics_provider_factory.statistics_provider_factory_ptr, thresholds_factory.thresholds_factory_ptr,
            rule_induction.rule_induction_ptr, default_rule_head_refinement_factory.head_refinement_factory_ptr,
            head_refinement_factory.head_refinement_factory_ptr,
            instance_sub_sampling_factory.instance_sub_sampling_factory_ptr,
            feature_sub_sampling_factory.feature_sub_sampling_factory_ptr,
            partition_sampling_factory.partition_sampling_factory_ptr, move(stopping_criteria_ptr))
