/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/vector_dense.hpp"
#include "common/indices/index_vector.hpp"
#include <memory>

// Forward declarations
class IStatistics;
class IHead;


/**
 * An abstract base class for all classes that store the scores that are predicted by a rule.
 */
class AbstractPrediction : public IIndexVector {

    private:

        DenseVector<float64> predictedScoreVector_;

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        AbstractPrediction(uint32 numElements);

        /**
         * An iterator that provides access to the predicted scores and allows to modify them.
         */
        typedef DenseVector<float64>::iterator score_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores.
         */
        typedef DenseVector<float64>::const_iterator score_const_iterator;

        /**
         * Returns a `score_iterator` to the beginning of the predicted scores.
         *
         * @return A `score_iterator` to the beginning
         */
        score_iterator scores_begin();

        /**
         * Returns a `score_iterator` to the end of the predicted scores.
         *
         * @return A `score_iterator` to the end
         */
        score_iterator scores_end();

        /**
         * Returns a `score_const_iterator` to the beginning of the predicted scores.
         *
         * @return A `score_const_iterator` to the beginning
         */
        score_const_iterator scores_cbegin() const;

        /**
         * Returns a `score_const_iterator` to the end of the predicted scores.
         *
         * @return A `score_const_iterator` to the end
         */
        score_const_iterator scores_cend() const;

        /**
         * Sets the predicted scores in another vector to this vector.
         *
         * @param begin A `score_const_iterator` to the beginning of the predicted scores
         * @param end   A `score_const_iterator` to the end of the predicted scores
         */
        void set(score_const_iterator begin, score_const_iterator end);

        /**
         * Creates and returns a head that contains the scores that are stored by this prediction.
         *
         * @return An unique pointer to an object of type `IHead` that has been created
         */
        virtual std::unique_ptr<IHead> toHead() const = 0;

        /**
         * Sets the number of labels for which the rule predict.
         *
         * @param numElements   The number of labels to be set
         * @param freeMemory    True, if unused memory should be freed if possible, false otherwise
         */
        virtual void setNumElements(uint32 numElements, bool freeMemory);

        uint32 getNumElements() const override;

};
