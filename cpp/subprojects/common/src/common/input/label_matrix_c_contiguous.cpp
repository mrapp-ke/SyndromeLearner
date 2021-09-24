#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/statistics/statistics_provider_factory.hpp"
#include "common/sampling/partition_sampling.hpp"
#include "common/sampling/instance_sampling.hpp"


CContiguousLabelMatrix::CContiguousLabelMatrix(uint32 numRows, uint32 numCols, const uint32* array)
    : timeSlots_(DenseVector<uint32>(numRows)), indices_(DenseVector<uint32>(numRows)),
      values_(DenseVector<uint32>(numRows)) {
    DenseVector<uint32>::iterator timeSlotIterator = timeSlots_.begin();
    DenseVector<uint32>::iterator indexIterator = indices_.begin();
    DenseVector<uint32>::iterator valueIterator = values_.begin();
    uint32 exampleIndex = 0;
    uint32 timeSlotIndex = 0;
    timeSlotIterator[exampleIndex] = timeSlotIndex;
    indexIterator[timeSlotIndex] = exampleIndex;
    valueIterator[timeSlotIndex] = array[1];
    uint32 previousTimestamp = array[0];

    for (exampleIndex = exampleIndex + 1; exampleIndex < numRows; exampleIndex++) {
        uint32 offset = exampleIndex * numCols;
        uint32 timestamp = array[offset];

        if (timestamp != previousTimestamp) {
            timeSlotIndex++;
            indexIterator[timeSlotIndex] = exampleIndex;
            valueIterator[timeSlotIndex] = array[offset + 1];
            previousTimestamp = timestamp;
        }

        timeSlotIterator[exampleIndex] = timeSlotIndex;
    }

    uint32 numTimeSlots = timeSlotIndex + 1;
    indexIterator[timeSlotIndex + 1] = numRows;
    indices_.setNumElements(numTimeSlots + 1, true);
    values_.setNumElements(numTimeSlots, true);
}

CContiguousLabelMatrix::index_const_iterator CContiguousLabelMatrix::time_slots_cbegin() const {
    return timeSlots_.cbegin();
}

CContiguousLabelMatrix::index_const_iterator CContiguousLabelMatrix::time_slots_cend() const {
    return timeSlots_.cend();
}

CContiguousLabelMatrix::value_const_iterator CContiguousLabelMatrix::values_cbegin() const {
    return values_.cbegin();
}

CContiguousLabelMatrix::value_const_iterator CContiguousLabelMatrix::values_cend() const {
    return values_.cend();
}

CContiguousLabelMatrix::index_const_iterator CContiguousLabelMatrix::indices_cbegin() const {
    return indices_.cbegin();
}

CContiguousLabelMatrix::index_const_iterator CContiguousLabelMatrix::indices_cend() const {
    return indices_.cend();
}

uint32 CContiguousLabelMatrix::getNumRows() const {
    return timeSlots_.getNumElements();
}

uint32 CContiguousLabelMatrix::getNumCols() const {
    return 1;
}

uint32 CContiguousLabelMatrix::getNumTimeSlots() const {
    return values_.getNumElements();
}

std::unique_ptr<IStatisticsProvider> CContiguousLabelMatrix::createStatisticsProvider(
        const IStatisticsProviderFactory& factory) const {
    return factory.create(*this);
}

std::unique_ptr<IPartitionSampling> CContiguousLabelMatrix::createPartitionSampling(
        const IPartitionSamplingFactory& factory) const {
    return factory.create(*this);
}
