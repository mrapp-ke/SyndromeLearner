/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics_immutable.hpp"
#include "common/head_refinement/prediction_full.hpp"
#include "common/head_refinement/prediction_partial.hpp"
#include <vector>


/**
 * Defines an interface for all classes that inherit from `IImmutableStatistics`, but do also provide functions that
 * allow to only use a sub-sample of the available statistics, as well as to update the statistics after a new rule has
 * been learned.
 */
class IStatistics : virtual public IImmutableStatistics {

    public:

        virtual ~IStatistics() { };

        /**
         * Resets the statistics which should be considered in the following for learning a new rule. The indices of the
         * respective statistics must be provided via subsequent calls to the function `addSampledStatistic`.
         *
         * This function must be invoked before a new rule is learned from scratch, as each rule may be learned on a
         * different sub-sample of the statistics.
         *
         * This function is supposed to reset any non-global internal state that only holds for a certain subset of the
         * available statistics and therefore becomes invalid when a different subset of the statistics should be used.
         */
        virtual void resetSampledStatistics() = 0;

        /**
         * Adds a specific statistic to the sub-sample that should be considered in the following for learning a new
         * rule from scratch.
         *
         * This function must be called repeatedly for each statistic that should be considered, immediately after the
         * invocation of the function `resetSampledStatistics`.
         *
         * This function is supposed to update any internal state that relates to the considered statistics, i.e., to
         * compute and store local information that is required by the other function that will be called later. Any
         * information computed by this function is expected to be reset when invoking the function
         * `resetSampledStatistics` for the next time.
         *
         * @param statisticIndex    The index of the statistic that should be considered
         * @param weight            The weight of the statistic that should be considered
         */
        virtual void addSampledStatistic(uint32 statisticIndex, float64 weight) = 0;

        /**
         * Resets the statistics which should be considered in the following for refining an existing rule. The indices
         * of the respective statistics must be provided via subsequent calls to the function `updateCoveredStatistic`.
         *
         * This function must be invoked each time an existing rule has been refined, i.e., when a new condition has
         * been added to its body, because this results in a subset of the statistics being covered by the refined rule.
         *
         * This function is supposed to reset any non-global internal state that only holds for a certain subset of the
         * available statistics and therefore becomes invalid when a different subset of the statistics should be used.
         */
        virtual void resetCoveredStatistics() = 0;

        /**
         * Adds a specific statistic to the subset that is covered by an existing rule and therefore should be
         * considered in the following for refining an existing rule.
         *
         * This function must be called repeatedly for each statistic that is covered by the existing rule, immediately
         * after the invocation of the function `resetCoveredStatistics`.
         *
         * Alternatively, this function may be used to indicate that a statistic, which has previously been passed to
         * this function, should not be considered anymore by setting the argument `remove` accordingly.
         *
         * This function is supposed to update any internal state that relates to the considered statistics, i.e., to
         * compute and store local information that is required by the other function that will be called later. Any
         * information computed by this function is expected to be reset when invoking the function
         * `resetCoveredStatistics` for the next time.
         *
         * @param statisticIndex    The index of the statistic that should be updated
         * @param weight            The weight of the statistic that should be updated
         * @param remove            False, if the statistic should be considered, True, if the statistic should not be
         *                          considered anymore
         */
        virtual void updateCoveredStatistic(uint32 statisticIndex, float64 weight, bool remove) = 0;

        virtual void increaseCoverageCount(uint32 statisticIndex) = 0;

        virtual void updatePredictions() = 0;

        virtual std::unique_ptr<std::vector<uint32>> getGroundTruth() const = 0;

        virtual std::unique_ptr<std::vector<uint32>> getPredictions() const = 0;

};
