/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <memory>

// Forward declarations
class IStatisticsProvider;
class IStatisticsProviderFactory;
class IPartitionSampling;
class IPartitionSamplingFactory;
class SinglePartition;
class BiPartition;


/**
 * Defines an interface for all label matrices that provide access to the labels of the training examples.
 */
class ILabelMatrix {

    public:

        virtual ~ILabelMatrix() { };

        /**
         * Returns the number of available examples.
         *
         * @return The number of examples
         */
        virtual uint32 getNumRows() const = 0;

        /**
         * Returns the number of available labels.
         *
         * @return The number of labels
         */
        virtual uint32 getNumCols() const = 0;

        /**
         * Returns the number of available time slots.
         *
         * @return The number of time slots
         */
        virtual uint32 getNumTimeSlots() const = 0;

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`, based on the type of this label
         * matrix.
         *
         * @param factory   A reference to an object of type `IStatisticsProviderFactory` that should be used to create
         *                  the instance
         * @return          An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
            const IStatisticsProviderFactory& factory) const = 0;

        /**
         * Creates and returns a new instance of the class `IPartitionSampling`, based on the type of this label matrix.
         *
         * @param factory   A reference to an object of type `IPartitionSamplingFactory` that should be used to create
         *                  the instance
         * @return          An unique pointer to an object of type `IPartitionSampling` that has been created
         */
        virtual std::unique_ptr<IPartitionSampling> createPartitionSampling(
            const IPartitionSamplingFactory& factory) const = 0;

};
