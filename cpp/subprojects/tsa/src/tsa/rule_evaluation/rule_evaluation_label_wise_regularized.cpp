#include "tsa/rule_evaluation/rule_evaluation_label_wise_regularized.hpp"
#include "common/rule_evaluation/score_vector_label_wise_dense.hpp"
#include <cmath>


namespace tsa {

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
     * Hessians that have been calculated according to a loss function that is applied label-wise using L2
     * regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<class T>
    class RegularizedLabelWiseRuleEvaluation final : public ILabelWiseRuleEvaluation {

        private:

            DenseLabelWiseScoreVector<T> scoreVector_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             */
            RegularizedLabelWiseRuleEvaluation(const T& labelIndices)
                : scoreVector_(DenseLabelWiseScoreVector<T>(labelIndices)) {
                typename DenseLabelWiseScoreVector<T>::score_iterator iterator = scoreVector_.scores_begin();

                for (uint32 i = 0; i < scoreVector_.getNumElements(); i++) {
                    iterator[i] = 1.0;
                }
            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const DenseVector<uint32>& predictions, const CContiguousLabelMatrix& groundTruth) override {
                uint32 numTimeSlots = predictions.getNumElements();
                CContiguousLabelMatrix::value_const_iterator groundTruthIterator = groundTruth.values_cbegin();
                DenseVector<uint32>::const_iterator predictionIterator = predictions.cbegin();
                float64 xSum = 0;
                float64 xSquaredSum = 0;
                float64 ySum = 0;
                float64 ySquaredSum = 0;
                float64 productSum = 0;

                for (uint32 i = 0; i < numTimeSlots; i++) {
                    float64 x = (float64) groundTruthIterator[i];
                    float64 y = (float64) predictionIterator[i];
                    xSum += x;
                    xSquaredSum += std::pow(x, 2);
                    ySum += y;
                    ySquaredSum += std::pow(y, 2);
                    float64 product = x * y;
                    productSum += product;
                }

                float64 n = (float64) numTimeSlots;
                float64 numerator = (n * productSum) - (xSum * ySum);
                float64 sqrt1 = std::sqrt((n * xSquaredSum) - std::pow(xSum, 2));
                float64 sqrt2 = std::sqrt((n * ySquaredSum) - std::pow(ySum, 2));
                float64 denominator = sqrt1 * sqrt2;

                scoreVector_.overallQualityScore = -std::abs(numerator / denominator);

                return scoreVector_;
            }

    };

    std::unique_ptr<ILabelWiseRuleEvaluation> RegularizedLabelWiseRuleEvaluationFactory::create(
            const FullIndexVector& indexVector) const {
        return std::make_unique<RegularizedLabelWiseRuleEvaluation<FullIndexVector>>(indexVector);
    }

    std::unique_ptr<ILabelWiseRuleEvaluation> RegularizedLabelWiseRuleEvaluationFactory::create(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<RegularizedLabelWiseRuleEvaluation<PartialIndexVector>>(indexVector);
    }

}
