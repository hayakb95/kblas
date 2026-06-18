// ============================================================================
// Matrix.hpp - Matrix class declaration
// ============================================================================
#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>

template <typename T>
class Matrix {
private:
    std::vector<T> data;  // Single contiguous vector (row-major order)
    size_t rows;
    size_t cols;

public:
    // Constructors
    Matrix(size_t r, size_t c);
    Matrix(size_t r, size_t c, const T& initial);

    // Access elements
    T& operator()(size_t row, size_t col);
    const T& operator()(size_t row, size_t col) const;

    // Get dimensions
    size_t getRows() const;
    size_t getCols() const;

    // Direct data access (useful for optimization)
    T* getData();
    const T* getData() const;

    // Matrix operations
    Matrix<T> operator+(const Matrix<T>& other) const;
    Matrix<T> operator-(const Matrix<T>& other) const;
    Matrix<T> operator*(const Matrix<T>& other) const;
    Matrix<T> operator*(const T& scalar) const;
    Matrix<T> transpose() const;
    Matrix<T> optTranspose(size_t bsize_row = 1, size_t bsize_col = 1) const;
    Matrix<T> frobeniusNorm() const;

    // Utility
    void print() const;
};

// Include the template implementation
#include "Matrix.tpp"

#endif // MATRIX_HPP

