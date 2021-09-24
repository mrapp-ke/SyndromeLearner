#include "tsa/statistics/statistics_label_wise_dense.hpp"
#include "tsa/data/matrix_dense_numeric.hpp"
#include "statistics_label_wise_common.hpp"
#include "omp.h"


namespace tsa {

    template<class LabelMatrix>
    static inline std::unique_ptr<ILabelWiseStatistics> createInternally(
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, const LabelMatrix& labelMatrix) {
        return std::make_unique<LabelWiseStatistics<LabelMatrix>>(ruleEvaluationFactoryPtr, labelMatrix);
    }

    DenseLabelWiseStatisticsFactory::DenseLabelWiseStatisticsFactory(
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr)
        : ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr) {

    }

    std::unique_ptr<ILabelWiseStatistics> DenseLabelWiseStatisticsFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        return createInternally<CContiguousLabelMatrix>(ruleEvaluationFactoryPtr_, labelMatrix);
    }

}
