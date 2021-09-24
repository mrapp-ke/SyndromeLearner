from rl.common.cython.model cimport IModelBuilder, ModelBuilder


cdef extern from "tsa/model/rule_list.hpp" namespace "tsa" nogil:

    cdef cppclass RuleListBuilderImpl"tsa::RuleListBuilder"(IModelBuilder):
        pass


cdef class RuleListBuilder(ModelBuilder):
    pass
