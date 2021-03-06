/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/head_refinement/head_refinement_factory.hpp"
#include "common/model/model_builder.hpp"
#include "common/sampling/feature_sampling.hpp"
#include "common/sampling/weight_vector.hpp"
#include "common/sampling/partition.hpp"
#include "common/statistics/statistics_provider.hpp"
#include "common/thresholds/thresholds.hpp"
#include <utility>



/**
 * Defines an interface for all classes that implement an algorithm for inducing individual rules.
 */
class IRuleInduction {

    public:

        virtual ~IRuleInduction() { };

        /**
         * Induces the default rule.
         *
         * @param statisticsProvider    A reference to an object of type `IStatisticsProvider` that provides access to
         *                              the statistics which should serve as the basis for inducing the default rule
         * @param headRefinementFactory A pointer to an object of type `IHeadRefinementFactory` that allows to create
         *                              instances of the class that should be used to find the head of the default rule
         *                              or a null pointer, if no default rule should be induced
         * @param modelBuilder          A reference to an object of type `IModelBuilder`, the default rule should be
         *                              added to
         */
        virtual void induceDefaultRule(IStatisticsProvider& statisticsProvider,
                                       const IHeadRefinementFactory* headRefinementFactory,
                                       IModelBuilder& modelBuilder) const = 0;

        /**
         * Induces a new rule.
         *
         * @param thresholds                A reference to an object of type `IThresholds` that provides access to the
         *                                  thresholds that may be used by the conditions of the rule
         * @param labelIndices              A reference to an object of type `IIndexVector` that provides access to the
         *                                  indices of the labels for which the rule may predict
         * @param weights                   A reference to an object of type `IWeightVector` that provides access to the
         *                                  weights of individual training examples
         * @param partition                 A reference to an object of type `IPartition` that provides access to the
         *                                  indices of the training examples that belong to the training set and the
         *                                  holdout set, respectively
         * @param featureSubSampling        A reference to an object of type `IFeatureSubSampling` that should be used
         *                                  for sampling the features that may be used by a new condition
         * @param rng                       A reference to an object of type `RNG` that implements the random number
         *                                  generator to be used
         * @param modelBuilder              A reference to an object of type `IModelBuilder`, the rule should be added
         *                                  to
         * @return                          True, if a rule has been induced, false otherwise
         */
        virtual std::pair<bool, float64> induceRule(IThresholds& thresholds, const IIndexVector& labelIndices,
                                const IWeightVector& weights, IPartition& partition,
                                IFeatureSubSampling& featureSubSampling, RNG& rng, IModelBuilder& modelBuilder,
                                float64 currentQuality) const = 0;

};
