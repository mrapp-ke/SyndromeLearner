"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class HeadRefinementFactory:
    """
    A wrapper for the pure virtual C++ class `IHeadRefinementFactory`.
    """
    pass


cdef class NoHeadRefinementRefinementFactory(HeadRefinementFactory):
    """
    A `HeadRefinementFactory` that stores a null pointer.
    """
    pass


cdef class FullHeadRefinementFactory(HeadRefinementFactory):
    """
    A wrapper for the C++ class `FullHeadRefinementFactory`.
    """

    def __cinit__(self):
        self.head_refinement_factory_ptr = <shared_ptr[IHeadRefinementFactory]>make_shared[FullHeadRefinementFactoryImpl]()
