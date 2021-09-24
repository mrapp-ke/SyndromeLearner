from rl.common.cython.statistics cimport StatisticsProviderFactory, IStatisticsProviderFactory
from rl.tsa.cython.rule_evaluation_label_wise cimport ILabelWiseRuleEvaluationFactory

from libcpp.memory cimport shared_ptr


cdef extern from "tsa/statistics/statistics_label_wise_provider.hpp" namespace "tsa" nogil:

    cdef cppclass LabelWiseStatisticsProviderFactoryImpl"tsa::LabelWiseStatisticsProviderFactory"(
            IStatisticsProviderFactory):

        # Constructors:

        LabelWiseStatisticsProviderFactoryImpl(
            shared_ptr[ILabelWiseRuleEvaluationFactory] defaultRuleEvaluationFactoryPtr,
            shared_ptr[ILabelWiseRuleEvaluationFactory] ruleEvaluationFactoryPtr) except +


cdef class LabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    pass
