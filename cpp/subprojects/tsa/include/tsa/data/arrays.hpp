/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include <algorithm>


namespace tsa {

    /**
     * Sets all elements in an array `a` to the difference between the elements in two other arrays `b` and `c`, such
     * that `a = b - c`.
     *
     * @tparam T            The type of the arrays `a`, `b` and `c`
     * @param a             A pointer to an array of template type `T` to be updated
     * @param b             A pointer to an array of template type `T`
     * @param c             A pointer to an array of template type `T`
     * @param numElements   The number of elements in the arrays `a`, `b` and `c`
     */
    template<class T>
    static inline void setArrayToDifference(T* a, const T* b, const T* c, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] = b[i] - c[i];
        }
    }

    /**
     * Sets all elements in an array `a` to the difference between the elements in two other array `b` and `c`, such
     * that `a = b - c`. The indices of elements in the arrays `b` and `c` that correspond to the elements in array `a`
     * are given as an additional array.
     *
     * @tparam T            The type of the arrays `a`, `b` and `c`
     * @param a             A pointer to an array of template type `T` to be updated
     * @param b             A pointer to an array of template type `T`
     * @param c             A pointer to an array of template type `T`
     * @param indices       A pointer to an array of type `uint32` that stores the indices of the elements in the arrays
     *                      `a` and `b` that correspond to the elements in array `a`
     * @param numElements   The number of elements in the array `a`
     */
    template<class T>
    static inline void setArrayToDifference(T* a, const T* b, const T* c, const uint32* indices, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] = b[index] - c[i];
        }
    }

    /**
     * Adds the elements in an array `b` to the elements in another array `b`, such that `a = a + b`.
     *
     * @tparam T            The type of the arrays `a` and `b`
     * @param a             A pointer to an array of template type `T` to be updated
     * @param b             A pointer to an array of template type `T`
     * @param numElements   The number of elements in the arrays `a` and `b`
     *
     */
    template<typename T>
    static inline void addToArray(T* a, const T* b, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] += b[i];
        }
    }

    /**
     * Adds the elements in an array `b` to the elements in another array `b`. The elements in the array `b` are
     * multiplied by a given weight, such that `a = a + (b * weight)`.
     *
     * @tparam T            The type of the arrays `a` and `b`
     * @param a             A pointer to an array of template type `T` to be updated
     * @param b             A pointer to an array of template type `T`
     * @param numElements   The number of elements in the arrays `a` and `b`
     * @param weight        The weight, the elements in the array `b` should be multiplied by
     *
     */
    template<typename T>
    static inline void addToArray(T* a, const T* b, uint32 numElements, T weight) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] += (b[i] * weight);
        }
    }

    /**
     * Adds the elements in an array `b` to the elements in another array `b`. The elements in the array `b` are
     * multiplied by a given weight, such that `a = a + (b * weight)`. The indices of elements in the array `b` that
     * correspond to the elements in array `a` are given as an additional array.
     *
     * @tparam T            The type of the arrays `a` and `b`
     * @param a             A pointer to an array of template type `T` to be updated
     * @param b             A pointer to an array of template type `T`
     * @param numElements   The number of elements in the arrays `a` and `b`
     * @param weight        The weight, the elements in the array `b` should be multiplied by
     * @param indices       A pointer to an array of type `uint32` that stores the indices of the elements in the array
     *                      `b` that correspond to the elements in array `a`
     *
     */
    template<class T>label
    static inline void addToArray(T* a, const T* b, const uint32* indices, uint32 numElements, T weight) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] += (b[index] * weight);
        }
    }

}
