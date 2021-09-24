#include "common/rule_induction/rule_induction_top_down.hpp"
#include "common/indices/index_vector_full.hpp"
#include "omp.h"
#include <unordered_map>


TopDownRuleInduction::TopDownRuleInduction(float32 minSupport, intp maxConditions, uint32 numThreads)
    : minSupport_(minSupport), maxConditions_(maxConditions), numThreads_(numThreads) {

}

void TopDownRuleInduction::induceDefaultRule(IStatisticsProvider& statisticsProvider,
                                             const IHeadRefinementFactory* headRefinementFactory,
                                             IModelBuilder& modelBuilder) const {
    statisticsProvider.switchRuleEvaluation();
}

std::pair<bool, float64> TopDownRuleInduction::induceRule(IThresholds& thresholds, const IIndexVector& labelIndices,
                                                          const IWeightVector& weights, IPartition& partition,
                                                          IFeatureSubSampling& featureSubSampling, RNG& rng,
                                                          IModelBuilder& modelBuilder, float64 currentQuality) const {
    uint32 numExamples = thresholds.getNumExamples();
    uint32 minCoverage = (uint32) (minSupport_ * numExamples);
    // The label indices for which the next refinement of the rule may predict
    const IIndexVector* currentLabelIndices = &labelIndices;
    // A (stack-allocated) list that contains the conditions in the rule's body (in the order they have been learned)
    ConditionList conditions;
    // The total number of conditions
    uint32 numConditions = 0;
    // A map that stores a pointer to an object of type `IRuleRefinement` for each feature
    std::unordered_map<uint32, std::unique_ptr<IRuleRefinement>> ruleRefinements;
    std::unordered_map<uint32, std::unique_ptr<IRuleRefinement>>* ruleRefinementsPtr = &ruleRefinements;
    // An unique pointer to the best refinement of the current rule
    std::unique_ptr<Refinement> bestRefinementPtr = std::make_unique<Refinement>();
    // A pointer to the head of the best rule found so far
    AbstractEvaluatedPrediction* bestHead = nullptr;
    // Whether a refinement of the current rule has been found
    bool foundRefinement = true;

    // Create a new subset of the given thresholds...
    std::unique_ptr<IThresholdsSubset> thresholdsSubsetPtr = thresholds.createSubset(weights);

    // Search for the best refinement until no improvement in terms of the rule's quality score is possible anymore or
    // the maximum number of conditions has been reached...
    while (foundRefinement && (maxConditions_ == -1 || numConditions < maxConditions_)) {
        foundRefinement = false;

        // Sample features...
        const IIndexVector& sampledFeatureIndices = featureSubSampling.subSample(rng);
        uint32 numSampledFeatures = sampledFeatureIndices.getNumElements();

        // For each feature, create an object of type `IRuleRefinement`...
        for (intp i = 0; i < numSampledFeatures; i++) {
            uint32 featureIndex = sampledFeatureIndices.getIndex((uint32) i);
            std::unique_ptr<IRuleRefinement> ruleRefinementPtr = currentLabelIndices->createRuleRefinement(
                *thresholdsSubsetPtr, featureIndex);
            ruleRefinements[featureIndex] = std::move(ruleRefinementPtr);
        }

        // Search for the best condition among all available features to be added to the current rule...
        #pragma omp parallel for firstprivate(numSampledFeatures) firstprivate(ruleRefinementsPtr) \
        firstprivate(bestHead) schedule(dynamic) num_threads(numThreads_)
        for (intp i = 0; i < numSampledFeatures; i++) {
            uint32 featureIndex = sampledFeatureIndices.getIndex((uint32) i);
            std::unique_ptr<IRuleRefinement>& ruleRefinementPtr = ruleRefinementsPtr->find(featureIndex)->second;
            ruleRefinementPtr->findRefinement(bestHead, minCoverage);
        }

        // Pick the best refinement among the refinements that have been found for the different features...
        for (intp i = 0; i < numSampledFeatures; i++) {
            uint32 featureIndex = sampledFeatureIndices.getIndex((uint32) i);
            std::unique_ptr<IRuleRefinement>& ruleRefinementPtr = ruleRefinements.find(featureIndex)->second;
            std::unique_ptr<Refinement> refinementPtr = ruleRefinementPtr->pollRefinement();

            if (refinementPtr->isBetterThan(*bestRefinementPtr)) {
                bestRefinementPtr = std::move(refinementPtr);
                foundRefinement = true;
            }
        }

        if (foundRefinement) {
            bestHead = bestRefinementPtr->headPtr.get();

            // Filter the current subset of thresholds by applying the best refinement that has been found...
            thresholdsSubsetPtr->filterThresholds(*bestRefinementPtr);

            // Add the new condition...
            conditions.addCondition(*bestRefinementPtr);
            numConditions++;
        }
    }

    if (bestHead == nullptr) {
        // No rule could be induced, because no useful condition could be found. This might be the case, if all examples
        // have the same values for the considered features.
        return std::make_pair(false, currentQuality);
    } else {
        float64 qualityScore = bestHead->overallQualityScore;

        if (qualityScore < currentQuality) {
            // Update the statistics by applying the predictions of the new rule...
            thresholdsSubsetPtr->applyPrediction(*bestHead);

            // Add the induced rule to the model...
            modelBuilder.addRule(conditions, *bestHead);
            return std::make_pair(true, qualityScore);
        } else {
            return std::make_pair(false, currentQuality);
        }
    }
}
