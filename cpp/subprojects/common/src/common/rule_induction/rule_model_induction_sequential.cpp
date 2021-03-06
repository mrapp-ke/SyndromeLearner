#include "common/rule_induction/rule_model_induction_sequential.hpp"


static inline IStoppingCriterion::Result testStoppingCriteria(
        std::forward_list<std::shared_ptr<IStoppingCriterion>>& stoppingCriteria, const IPartition& partition,
        const IStatistics& statistics, uint32 numRules) {
    IStoppingCriterion::Result result;
    result.action = IStoppingCriterion::Action::CONTINUE;

    for (auto it = stoppingCriteria.begin(); it != stoppingCriteria.end(); it++) {
        std::shared_ptr<IStoppingCriterion>& stoppingCriterionPtr = *it;
        IStoppingCriterion::Result stoppingCriterionResult = stoppingCriterionPtr->test(partition, statistics,
                                                                                        numRules);
        IStoppingCriterion::Action action = stoppingCriterionResult.action;

        switch (action) {
            case IStoppingCriterion::Action::FORCE_STOP: {
                result.action = action;
                result.numRules = stoppingCriterionResult.numRules;
                return result;
            }
            case IStoppingCriterion::Action::STORE_STOP: {
                result.action = action;
                result.numRules = stoppingCriterionResult.numRules;
                break;
            }
            default: {
                break;
            }
        }
    }

    return result;
}

SequentialRuleModelInduction::SequentialRuleModelInduction(
        std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
        std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr, std::shared_ptr<IRuleInduction> ruleInductionPtr,
        std::shared_ptr<IHeadRefinementFactory> defaultRuleHeadRefinementFactoryPtr,
        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
        std::shared_ptr<IInstanceSubSamplingFactory> instanceSubSamplingFactoryPtr,
        std::shared_ptr<IFeatureSubSamplingFactory> featureSubSamplingFactoryPtr,
        std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
        std::unique_ptr<std::forward_list<std::shared_ptr<IStoppingCriterion>>> stoppingCriteriaPtr)
    : statisticsProviderFactoryPtr_(statisticsProviderFactoryPtr), thresholdsFactoryPtr_(thresholdsFactoryPtr),
      ruleInductionPtr_(ruleInductionPtr), defaultRuleHeadRefinementFactoryPtr_(defaultRuleHeadRefinementFactoryPtr),
      headRefinementFactoryPtr_(headRefinementFactoryPtr),
      instanceSubSamplingFactoryPtr_(instanceSubSamplingFactoryPtr),
      featureSubSamplingFactoryPtr_(featureSubSamplingFactoryPtr),
      partitionSamplingFactoryPtr_(partitionSamplingFactoryPtr),
      stoppingCriteriaPtr_(std::move(stoppingCriteriaPtr)) {

}

std::unique_ptr<RuleModel> SequentialRuleModelInduction::induceRules(
        std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr, std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
        std::shared_ptr<ILabelMatrix> labelMatrixPtr, RNG& rng, IModelBuilder& modelBuilder,
        PredictionVisitor groundTruthVisitor, PredictionVisitor predictionVisitor) {
    // Induce default rule...
    const IHeadRefinementFactory* defaultRuleHeadRefinementFactory = defaultRuleHeadRefinementFactoryPtr_.get();
    uint32 numRules = defaultRuleHeadRefinementFactory != nullptr ? 1 : 0;
    uint32 numUsedRules = 0;
    std::shared_ptr<IStatisticsProvider> statisticsProviderPtr =
        labelMatrixPtr->createStatisticsProvider(*statisticsProviderFactoryPtr_);
    ruleInductionPtr_->induceDefaultRule(*statisticsProviderPtr, defaultRuleHeadRefinementFactory, modelBuilder);

    // Induce the remaining rules...
    std::unique_ptr<IThresholds> thresholdsPtr = thresholdsFactoryPtr_->create(featureMatrixPtr, nominalFeatureMaskPtr,
                                                                               statisticsProviderPtr,
                                                                               headRefinementFactoryPtr_);
    uint32 numFeatures = thresholdsPtr->getNumFeatures();
    uint32 numLabels = thresholdsPtr->getNumLabels();
    std::unique_ptr<IPartitionSampling> partitionSamplingPtr = labelMatrixPtr->createPartitionSampling(
        *partitionSamplingFactoryPtr_);
    IPartition& partition = partitionSamplingPtr->partition(rng);
    std::unique_ptr<IInstanceSubSampling> instanceSubSamplingPtr = partition.createInstanceSubSampling(
        *instanceSubSamplingFactoryPtr_);
    std::unique_ptr<IFeatureSubSampling> featureSubSamplingPtr = featureSubSamplingFactoryPtr_->create(numFeatures);
    const FullIndexVector labelIndices(numLabels);
    IStoppingCriterion::Result stoppingCriterionResult;
    float64 currentQuality = 0;

    while (stoppingCriterionResult = testStoppingCriteria(*stoppingCriteriaPtr_, partition,
                                                          statisticsProviderPtr->get(), numRules),
           stoppingCriterionResult.action != IStoppingCriterion::Action::FORCE_STOP) {
        if (stoppingCriterionResult.action == IStoppingCriterion::Action::STORE_STOP && numUsedRules == 0) {
            numUsedRules = stoppingCriterionResult.numRules;
        }

        const IWeightVector& weights = instanceSubSamplingPtr->subSample(rng);
        std::pair<bool, float64> result = ruleInductionPtr_->induceRule(*thresholdsPtr, labelIndices, weights, partition,
                                                     *featureSubSamplingPtr, rng, modelBuilder, currentQuality);
        bool success = result.first;
        currentQuality = result.second;

        if (success) {
            numRules++;

            std::unique_ptr<std::vector<uint32>> predictionPtr = statisticsProviderPtr->get().getPredictions();
            predictionVisitor(*predictionPtr);
        } else {
            break;
        }
    }

    std::unique_ptr<std::vector<uint32>> groundTruthPtr = statisticsProviderPtr->get().getGroundTruth();
    groundTruthVisitor(*groundTruthPtr);

    // Build and return the final model...
    return modelBuilder.build(numUsedRules);
}
