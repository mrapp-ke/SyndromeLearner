#include "tsa/statistics/statistics_label_wise.hpp"
#include "common/statistics/statistics_subset_decomposable.hpp"
#include "common/data/arrays.hpp"


namespace tsa {

    /**
     * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied label-wise and allows to update the gradients and Hessians after a new rule has been learned.
     *
     * @tparam LabelMatrix      The type of the matrix that provides access to the labels of the training examples
     */
    template<class LabelMatrix>
    class LabelWiseStatistics final : virtual public ILabelWiseStatistics {

        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `LabelWiseStatistics`.
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<class T>
            class StatisticsSubset final : public AbstractDecomposableStatisticsSubset {

                private:

                    const LabelWiseStatistics& statistics_;

                    std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr_;

                    const T& labelIndices_;

                    DenseVector<uint32> coveredPredictionVector_;

                    DenseVector<uint32> uncoveredPredictionVector_;

                    DenseVector<uint32>* accumulatedCoveredPredictionVector_;

                    DenseVector<uint32>* accumulatedUncoveredPredictionVector_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `LabelWiseStatistics` that
                     *                          stores the gradients and Hessians
                     * @param ruleEvaluationPtr An unique pointer to an object of type `ILabelWiseRuleEvaluation` that
                     *                          should be used to calculate the predictions, as well as corresponding
                     *                          quality scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    StatisticsSubset(const LabelWiseStatistics& statistics,
                                     std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr, const T& labelIndices)
                        : statistics_(statistics), ruleEvaluationPtr_(std::move(ruleEvaluationPtr)),
                          labelIndices_(labelIndices),
                          coveredPredictionVector_(DenseVector<uint32>(statistics.predictionVector_)),
                          uncoveredPredictionVector_(DenseVector<uint32>(statistics.totalPredictionVector_)),
                          accumulatedCoveredPredictionVector_(nullptr), accumulatedUncoveredPredictionVector_(nullptr) {

                    }

                    ~StatisticsSubset() {
                        delete accumulatedCoveredPredictionVector_;
                        delete accumulatedUncoveredPredictionVector_;
                    }

                    void addToMissing(uint32 statisticIndex, float64 weight) override {
                        if (statistics_.coverageCountVector_[statisticIndex] == 0) {
                            uint32 timeSlot = statistics_.labelMatrix_.time_slots_cbegin()[statisticIndex];
                            uncoveredPredictionVector_[timeSlot] -= 1;
                        }
                    }

                    void addToSubset(uint32 statisticIndex, float64 weight) override {
                        if (statistics_.coverageCountVector_[statisticIndex] == 0) {
                            uint32 timeSlot = statistics_.labelMatrix_.time_slots_cbegin()[statisticIndex];
                            coveredPredictionVector_[timeSlot] += 1;
                            uncoveredPredictionVector_[timeSlot] -= 1;

                            if (accumulatedCoveredPredictionVector_ != nullptr) {
                                (*accumulatedCoveredPredictionVector_)[timeSlot] += 1;
                                (*accumulatedUncoveredPredictionVector_)[timeSlot] -= 1;
                            }
                        }
                    }

                    void resetSubset() override {
                        if (accumulatedCoveredPredictionVector_ == nullptr) {
                            accumulatedCoveredPredictionVector_ = new DenseVector<uint32>(coveredPredictionVector_);
                            accumulatedUncoveredPredictionVector_ = new DenseVector<uint32>(uncoveredPredictionVector_);
                        }

                        copyArray(statistics_.predictionVector_.cbegin(), coveredPredictionVector_.begin(),
                                  coveredPredictionVector_.getNumElements());
                        copyArray(statistics_.totalPredictionVector_.cbegin(), uncoveredPredictionVector_.begin(),
                                  uncoveredPredictionVector_.getNumElements());
                    }

                    const ILabelWiseScoreVector& calculateLabelWisePrediction(bool uncovered,
                                                                              bool accumulated) override {
                        const DenseVector<uint32>& predictions =
                            uncovered ? (accumulated ? *accumulatedUncoveredPredictionVector_ : uncoveredPredictionVector_)
                                      : (accumulated ? *accumulatedCoveredPredictionVector_ : coveredPredictionVector_);
                        return ruleEvaluationPtr_->calculateLabelWisePrediction(predictions, statistics_.labelMatrix_);
                    }

            };

            typedef StatisticsSubset<FullIndexVector> FullSubset;

            typedef StatisticsSubset<PartialIndexVector> PartialSubset;

            uint32 numStatistics_;

            uint32 numLabels_;

            DenseVector<uint32> coverageCountVector_;

            DenseVector<uint32> predictionVector_;

            DenseVector<uint32> totalPredictionVector_;

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            const LabelMatrix& labelMatrix_;

        public:

            /**
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`,
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             */
            LabelWiseStatistics(std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                const LabelMatrix& labelMatrix)
                : numStatistics_(labelMatrix.getNumRows()), numLabels_(labelMatrix.getNumCols()),
                  coverageCountVector_(DenseVector<uint32>(numStatistics_, true)),
                  predictionVector_(DenseVector<uint32>(labelMatrix.getNumTimeSlots(), true)),
                  totalPredictionVector_(DenseVector<uint32>(labelMatrix.getNumTimeSlots())),
                  ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr), labelMatrix_(labelMatrix) {

            }

            uint32 getNumStatistics() const override final {
                return numStatistics_;
            }

            uint32 getNumLabels() const override final {
                return numLabels_;
            }

            void setRuleEvaluationFactory(
                    std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) override {
                this->ruleEvaluationFactoryPtr_ = ruleEvaluationFactoryPtr;
            }

            void resetSampledStatistics() override {
                // This function is equivalent to the function `resetCoveredStatistics`...
                this->resetCoveredStatistics();
            }

            void addSampledStatistic(uint32 statisticIndex, float64 weight) override {
                // This function is equivalent to the function `updateCoveredStatistic`...
                this->updateCoveredStatistic(statisticIndex, weight, false);
            }

            void resetCoveredStatistics() override {
                copyArray(predictionVector_.cbegin(), totalPredictionVector_.begin(),
                          totalPredictionVector_.getNumElements());
            }

            void updateCoveredStatistic(uint32 statisticIndex, float64 weight, bool remove) override {
                if (coverageCountVector_[statisticIndex] == 0) {
                    uint32 timeSlot = labelMatrix_.time_slots_cbegin()[statisticIndex];

                    if (remove) {
                        totalPredictionVector_[timeSlot] -= 1;
                    } else {
                        totalPredictionVector_[timeSlot] += 1;
                    }
                }
            }

            void increaseCoverageCount(uint32 statisticIndex) override {
                coverageCountVector_[statisticIndex] += 1;
            }

            void updatePredictions() override {
                DenseVector<uint32>::iterator predictionIterator = predictionVector_.begin();
                DenseVector<uint32>::const_iterator countIterator = coverageCountVector_.cbegin();
                typename LabelMatrix::index_const_iterator indices = labelMatrix_.indices_cbegin();
                uint32 numTimeSlots = labelMatrix_.getNumTimeSlots();

                for (uint32 i = 0; i < numTimeSlots; i++) {
                    uint32 start = indices[i];
                    uint32 end = indices[i + 1];
                    uint32 prediction = 0;

                    for (uint32 j = start; j < end; j++) {
                        if (countIterator[j] > 0) {
                            prediction++;
                        }
                    }

                    predictionIterator[i] = prediction;
                }
            }

            std::unique_ptr<std::vector<uint32>> getGroundTruth() const override {
                std::unique_ptr<std::vector<uint32>> ptr = std::make_unique<std::vector<uint32>>();
                typename LabelMatrix::value_const_iterator groundTruthIterator = labelMatrix_.values_cbegin();
                uint32 numElements = labelMatrix_.getNumTimeSlots();

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 groundTruth = groundTruthIterator[i];
                    ptr->emplace_back(groundTruth);
                }

                return ptr;
            }

            std::unique_ptr<std::vector<uint32>> getPredictions() const override {
                std::unique_ptr<std::vector<uint32>> ptr = std::make_unique<std::vector<uint32>>();
                DenseVector<uint32>::const_iterator predictionIterator = predictionVector_.cbegin();
                uint32 numElements = predictionVector_.getNumElements();

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 prediction = predictionIterator[i];
                    ptr->emplace_back(prediction);
                }

                return ptr;
            }

            std::unique_ptr<IStatisticsSubset> createSubset(const FullIndexVector& labelIndices) const override {
                std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<typename LabelWiseStatistics::FullSubset>(*this, std::move(ruleEvaluationPtr),
                                                                                  labelIndices);
            }

            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices) const override {
                std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<typename LabelWiseStatistics::PartialSubset>(*this,
                                                                                     std::move(ruleEvaluationPtr),
                                                                                     labelIndices);
            }

    };

}
