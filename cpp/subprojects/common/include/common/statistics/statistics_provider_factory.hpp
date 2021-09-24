/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics_provider.hpp"
#include "common/input/label_matrix_c_contiguous.hpp"


/**
 * Defines an interface for all classes that allow to create instances of the class `IStatisticsProvider`.
 */
class IStatisticsProviderFactory {

    public:

        virtual ~IStatisticsProviderFactory() { };

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`, based on a label matrix that provides
         * random access to the labels of the training examples.
         *
         * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` that provides random access
         *                      to the labels of the training examples
         * @return              An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> create(const CContiguousLabelMatrix& labelMatrix) const = 0;

};
