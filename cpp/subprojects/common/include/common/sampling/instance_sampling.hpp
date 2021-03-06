/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/weight_vector.hpp"
#include "common/sampling/random.hpp"
#include <memory>

// Forward declarations
class BiPartition;
class SinglePartition;


/**
 * Defines an interface for all classes that implement a strategy for sub-sampling training examples.
 */
class IInstanceSubSampling {

    public:

        virtual ~IInstanceSubSampling() { };

        /**
         * Creates and returns a sub-sample of the available training examples.
         *
         * @param rng       A reference to an object of type `RNG`, implementing the random number generator to be used
         * @return          A reference to an object type `WeightVector` that provides access to the weights of the
         *                  individual training examples
         */
        virtual const IWeightVector& subSample(RNG& rng) = 0;

};


/**
 * Defines an interface for all factories that allow to create instances of the type `IInstanceSubSampling`.
 */
class IInstanceSubSamplingFactory {

    public:

        virtual ~IInstanceSubSamplingFactory() { };

        /**
         * Creates and returns a new object of type `IInstanceSubSampling`.
         *
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that are included in the training set
         * @return              An unique pointer to an object of type `IInstanceSubSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSubSampling> create(const SinglePartition& partition) const = 0;

        /**
         * Creates and returns a new object of type `IInstanceSubSampling`.
         *
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that are included in the training set and the holdout set,
         *                      respectively
         * @return              An unique pointer to an object of type `IInstanceSubSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSubSampling> create(BiPartition& partition) const = 0;

};
