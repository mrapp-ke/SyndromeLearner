/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_vector.hpp"


/**
 * An one-dimensional vector that provides random access to a fixed number of elements stored in a C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<class T>
class DenseVector final : public VectorView<T> {

    private:

        uint32 maxCapacity_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        DenseVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        DenseVector(uint32 numElements, bool init);

        DenseVector(const DenseVector<T>& vector);

        virtual ~DenseVector();

        /**
         * Sets the number of elements in the vector.
         *
         * @param numElements   The number of elements to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumElements(uint32 numElements, bool freeMemory);

};
