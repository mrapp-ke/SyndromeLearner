from rl.common.cython._types cimport uint32, intp
from rl.common.cython.input cimport NominalFeatureMask, INominalFeatureMask
from rl.common.cython.input cimport FeatureMatrix, IFeatureMatrix
from rl.common.cython.input cimport LabelMatrix, ILabelMatrix
from rl.common.cython.model cimport ModelBuilder, RuleModel, IModelBuilder, RuleModelImpl
from rl.common.cython.sampling cimport IInstanceSubSamplingFactory, IFeatureSubSamplingFactory, \
    IPartitionSamplingFactory, RNG
from rl.common.cython.statistics cimport IStatisticsProviderFactory
from rl.common.cython.stopping cimport IStoppingCriterion
from rl.common.cython.thresholds cimport IThresholdsFactory
from rl.common.cython.head_refinement cimport IHeadRefinementFactory
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.forward_list cimport forward_list
from libcpp.vector cimport vector


cdef extern from "common/rule_induction/rule_induction.hpp" nogil:

    cdef cppclass IRuleInduction:
        pass


ctypedef void (*PredictionVisitor)(const vector[uint32]&)


cdef extern from "common/rule_induction/rule_model_induction.hpp" nogil:

    cdef cppclass IRuleModelInduction:

        # Functions:

        unique_ptr[RuleModelImpl] induceRules(shared_ptr[INominalFeatureMask] nominalFeatureMaskPtr,
                                              shared_ptr[IFeatureMatrix] featureMatrixPtr,
                                              shared_ptr[ILabelMatrix] labelMatrixPtr, RNG& rng,
                                              IModelBuilder& modelBuilder, PredictionVisitor groundTruthVisitor,
                                              PredictionVisitor predictionVisitor)


cdef extern from "common/rule_induction/rule_induction_top_down.hpp" nogil:

    cdef cppclass TopDownRuleInductionImpl"TopDownRuleInduction"(IRuleInduction):

        # Constructors:

        TopDownRuleInductionImpl(uint32 minCoverage, intp maxConditions, uint32 numThreads) except +


cdef extern from "common/rule_induction/rule_model_induction_sequential.hpp" nogil:

    cdef cppclass SequentialRuleModelInductionImpl"SequentialRuleModelInduction"(IRuleModelInduction):

        # Constructors:

        SequentialRuleModelInductionImpl(
                shared_ptr[IStatisticsProviderFactory] statisticsProviderFactoryPtr,
                shared_ptr[IThresholdsFactory] thresholdsFactoryPtr, shared_ptr[IRuleInduction] ruleInductionPtr,
                shared_ptr[IHeadRefinementFactory] defaultRuleHeadRefinementFactoryPtr,
                shared_ptr[IHeadRefinementFactory] headRefinementFactoryPtr,
                shared_ptr[IInstanceSubSamplingFactory] instanceSubSamplingFactoryPtr,
                shared_ptr[IFeatureSubSamplingFactory] featureSubSamplingFactoryPtr,
                shared_ptr[IPartitionSamplingFactory] partitionSamplingFactoryPtr,
                unique_ptr[forward_list[shared_ptr[IStoppingCriterion]]] stoppingCriteriaPtr) except +


cdef extern from *:
    """
    #include "common/rule_induction/rule_model_induction.hpp"


    typedef void (*PredictionCythonVisitor)(void*, const std::vector<uint32>&);

    static inline IRuleModelInduction::PredictionVisitor wrapPredictionVisitor(void* self, PredictionCythonVisitor visitor) {
        return [=](const std::vector<uint32>& vec) {
            visitor(self, vec);
        };
    }
    """

    ctypedef void (*PredictionCythonVisitor)(void*, const vector[uint32]&)

    PredictionVisitor wrapPredictionVisitor(void* self, PredictionCythonVisitor visitor)


cdef class RuleInduction:

    # Attributes:

    cdef shared_ptr[IRuleInduction] rule_induction_ptr


cdef class TopDownRuleInduction(RuleInduction):
    pass


cdef class Predictions:

    # Attributes:

    cdef readonly list ground_truth

    cdef readonly list predictions


cdef class RuleModelInduction:

    # Attributes:

    cdef shared_ptr[IRuleModelInduction] rule_model_induction_ptr

    cdef readonly Predictions predictions

    # Functions:

    cdef __visit_ground_truth(self, const vector[uint32]& vec)

    cdef __visit_prediction(self, const vector[uint32]& vec)

    cpdef RuleModel induce_rules(self, NominalFeatureMask nominal_feature_mask, FeatureMatrix feature_matrix,
                                 LabelMatrix label_matrix, uint32 random_state, ModelBuilder model_builder)


cdef class SequentialRuleModelInduction(RuleModelInduction):
    pass
