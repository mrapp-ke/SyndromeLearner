/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics_provider_factory.hpp"
#include "tsa/statistics/statistics_label_wise.hpp"


namespace tsa {

    /**
     * Allows to create instances of the class `LabelWiseStatisticsProvider`.
     */
    class LabelWiseStatisticsProviderFactory : public IStatisticsProviderFactory {

        private:

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

        public:

            /**
             * @param defaultRuleEvaluationFactoryPtr   A shared pointer to an object of type
             *                                          `ILabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of the default rule
             * @param ruleEvaluationFactoryPtr          A shared pointer to an object of type
             *                                          `ILabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             */
            LabelWiseStatisticsProviderFactory(
                std::shared_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
                std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr);

            std::unique_ptr<IStatisticsProvider> create(const CContiguousLabelMatrix& labelMatrix) const override;

    };

}
