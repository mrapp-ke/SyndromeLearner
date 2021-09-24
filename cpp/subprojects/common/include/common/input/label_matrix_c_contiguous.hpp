/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/data/functions.hpp"
#include "common/input/label_matrix.hpp"
#include "common/data/vector_dense.hpp"


/**
 * Implements random read-only access to the labels of individual training examples that are stored in a pre-allocated
 * C-contiguous array.
 */
class CContiguousLabelMatrix final : public ILabelMatrix {

    private:

        DenseVector<uint32> timeSlots_;

        DenseVector<uint32> indices_;

        DenseVector<uint32> values_;

    public:

        /**
         * @param numRows   The number of rows in the label matrix
         * @param numCols   The number of columns in the label matrix
         * @param array     A pointer to a C-contiguous array of type `uint32` that stores the labels
         */
        CContiguousLabelMatrix(uint32 numRows, uint32 numCols, const uint32* array);

        /**
         * An iterator that provides read-only access to the values in the label matrix.
         */
        typedef CContiguousConstView<const uint32>::const_iterator value_const_iterator;

        typedef DenseVector<uint32>::const_iterator index_const_iterator;

        index_const_iterator time_slots_cbegin() const;

        index_const_iterator time_slots_cend() const;

        value_const_iterator values_cbegin() const;

        value_const_iterator values_cend() const;

        index_const_iterator indices_cbegin() const;

        index_const_iterator indices_cend() const;

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        uint32 getNumTimeSlots() const override;

        std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
            const IStatisticsProviderFactory& factory) const override;

        std::unique_ptr<IPartitionSampling> createPartitionSampling(
            const IPartitionSamplingFactory& factory) const override;

};
