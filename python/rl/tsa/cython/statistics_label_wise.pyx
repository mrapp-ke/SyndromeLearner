"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from rl.tsa.cython.rule_evaluation_label_wise cimport LabelWiseRuleEvaluationFactory

from libcpp.memory cimport make_shared


cdef class LabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    """
    A wrapper for the C++ class `LabelWiseStatisticsProviderFactory`.
    """

    def __cinit__(self, LabelWiseRuleEvaluationFactory default_rule_evaluation_factory,
                  LabelWiseRuleEvaluationFactory rule_evaluation_factory):
        """
        :param default_rule_evaluation_factory: The `LabelWiseRuleEvaluationFactory` that allows to create instances of
                                                the class that should be used for calculating the predictions, as well
                                                as corresponding quality scores, of the default rule
        :param rule_evaluation:                 The `LabelWiseRuleEvaluationFactory` that allows to create instances of
                                                the class that should be used for calculating the predictions, as well
                                                as corresponding quality scores, of rules
        """
        self.statistics_provider_factory_ptr = <shared_ptr[IStatisticsProviderFactory]>make_shared[LabelWiseStatisticsProviderFactoryImpl](
            default_rule_evaluation_factory.rule_evaluation_factory_ptr,
            rule_evaluation_factory.rule_evaluation_factory_ptr)
