from libcpp.memory cimport shared_ptr


cdef extern from "tsa/rule_evaluation/rule_evaluation_label_wise.hpp" namespace "tsa" nogil:

    cdef cppclass ILabelWiseRuleEvaluationFactory:
        pass


cdef extern from "tsa/rule_evaluation/rule_evaluation_label_wise_regularized.hpp" namespace "tsa" nogil:

    cdef cppclass RegularizedLabelWiseRuleEvaluationFactoryImpl"tsa::RegularizedLabelWiseRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):
        pass


cdef class LabelWiseRuleEvaluationFactory:

    # Attributes:

    cdef shared_ptr[ILabelWiseRuleEvaluationFactory] rule_evaluation_factory_ptr


cdef class RegularizedLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass
