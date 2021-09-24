/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_induction/rule_induction.hpp"


/**
 * Allows to induce classification rules using a top-down greedy search, where new conditions are added iteratively to
 * the (initially empty) body of a rule. At each iteration, the refinement that improves the rule the most is chosen.
 * The search stops if no refinement results in an improvement.
 */
class TopDownRuleInduction : public IRuleInduction {

    private:

        float32 minSupport_;

        intp maxConditions_;

        uint32 numThreads_;

    public:

        /**
         * @param minSupport                The minimum fraction of the training examples that must be covered by a
         *                                  rule. Must be in [0, 1)
         * @param maxConditions             The maximum number of conditions to be included in a rule's body. Must be at
         *                                  least 1 or -1, if the number of conditions should not be restricted
         * @param numThreads                The number of CPU threads to be used to search for potential refinements of
         *                                  a rule in parallel. Must be at least 1
         */
        TopDownRuleInduction(float32 minSupport, intp maxConditions, uint32 numThreads);

        void induceDefaultRule(IStatisticsProvider& statisticsProvider,
                               const IHeadRefinementFactory* headRefinementFactory,
                               IModelBuilder& modelBuilder) const override;

        std::pair<bool, float64> induceRule(IThresholds& thresholds, const IIndexVector& labelIndices,
                                            const IWeightVector& weights, IPartition& partition,
                                            IFeatureSubSampling& featureSubSampling, RNG& rng,
                                            IModelBuilder& modelBuilder, float64 currentQuality) const override;

};
