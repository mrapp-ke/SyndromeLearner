#include "common/sampling/partition_single.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/thresholds/thresholds_subset.hpp"
#include "common/rule_refinement/refinement.hpp"
#include "common/head_refinement/prediction.hpp"


SinglePartition::SinglePartition(uint32 numElements)
    : numElements_(numElements) {

}

SinglePartition::const_iterator SinglePartition::cbegin() const {
    return IndexIterator();
}

SinglePartition::const_iterator SinglePartition::cend() const {
    return IndexIterator(numElements_);
}

uint32 SinglePartition::getNumElements() const {
    return numElements_;
}

std::unique_ptr<IInstanceSubSampling> SinglePartition::createInstanceSubSampling(
        const IInstanceSubSamplingFactory& factory) {
    return factory.create(*this);
}

float64 SinglePartition::evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset,
                                             const ICoverageState& coverageState, const AbstractPrediction& head) {
    return coverageState.evaluateOutOfSample(thresholdsSubset, *this, head);
}

void SinglePartition::recalculatePrediction(const IThresholdsSubset& thresholdsSubset,
                                            const ICoverageState& coverageState, Refinement& refinement) {
    coverageState.recalculatePrediction(thresholdsSubset, *this, refinement);
}
