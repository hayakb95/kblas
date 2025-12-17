// ============================================================================
// Matrix.tpp - Matrix class implementation
// ============================================================================

// Constructors
template <typename T>
Matrix<T>::Matrix(size_t r, size_t c) : rows(r), cols(c) {
    data.resize(rows, std::vector<T>(cols, T()));
}

template <typename T>
Matrix<T>::Matrix(size_t r, size_t c, const T& initial) : rows(r), cols(c) {
    data.resize(rows, std::vector<T>(cols, initial));
}

// Access elements
template <typename T>
T& Matrix<T>::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row][col];
}

template <typename T>
const T& Matrix<T>::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row][col];
}

// Get dimensions
template <typename T>
size_t Matrix<T>::getRows() const {
    return rows;
}

template <typename T>
size_t Matrix<T>::getCols() const {
    return cols;
}

// Matrix addition
template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] + other(i, j);
        }
    }
    return result;
}

// Matrix subtraction
template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] - other(i, j);
        }
    }
    return result;
}

// Matrix multiplication
template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }
    Matrix<T> result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            T sum = T();
            for (size_t k = 0; k < cols; ++k) {
                sum += data[i][k] * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// Scalar multiplication
template <typename T>
Matrix<T> Matrix<T>::operator*(const T& scalar) const {
    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * scalar;
        }
    }
    return result;
}

// Transpose
template <typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = data[i][j];
        }
    }
    return result;
}

// Print matrix
template <typename T>
void Matrix<T>::print() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Optimized Transpose
// Write an optimized transpose function
template <typename T>
Matrix<T> Matrix<T>::optTranspose(size_t bsize_row, size_t bsize_col) const {
    size_t col_remainder = cols % bsize_col;
    size_t row_remainder = rows % bsize_row;

    size_t col_blocks = cols / bsize_col;
    size_t row_blocks = rows / bsize_row;

    Matrix<T> result(cols, rows); // Allocated transposed matrix

    for (size_t rBlockIdx = 0; rBlockIdx < row_blocks; ++rBlockIdx) {
        for (size_t cBlockIdx = 0; cBlockIdx < col_blocks; ++cBlockIdx) {
            for (size_t i = 0; i < bsize_row; ++i) {
                for (size_t j = 0; j < bsize_col; ++j) {
                    result(cBlockIdx * bsize_col + j, rBlockIdx * bsize_row + i) = data[rBlockIdx * bsize_row + i][cBlockIdx * bsize_col + j];
                }
            }
        }
    }

    // Handle remainder rows
    if (row_remainder > 0) {
        for (size_t cBlockIdx = 0; cBlockIdx < col_blocks; ++cBlockIdx) {
            for (size_t i = 0; i < row_remainder; ++i) {
                for (size_t j = 0; j < bsize_col; ++j) {
                    result(cBlockIdx * bsize_col + j, row_blocks * bsize_row + i) = data[row_blocks * bsize_row + i][cBlockIdx * bsize_col + j];
                }
            }
        }
    }

    // Handle remainder columns
    if (col_remainder > 0) {
        for (size_t rBlockIdx = 0; rBlockIdx < row_blocks; ++rBlockIdx) {
            for (size_t i = 0; i < bsize_row; ++i) {
                for (size_t j = 0; j < col_remainder; ++j) {
                    result(col_blocks * bsize_col + j, rBlockIdx * bsize_row + i) = data[rBlockIdx * bsize_row + i][col_blocks * bsize_col + j];
                }
            }
        }
    }

    // Handle bottom-right remainder block
    if (row_remainder > 0 && col_remainder > 0) {
        for (size_t i = 0; i < row_remainder; ++i) {
            for (size_t j = 0; j < col_remainder; ++j) {
                result(col_blocks * bsize_col + j, row_blocks * bsize_row + i) = data[row_blocks * bsize_row + i][col_blocks * bsize_col + j];
            }
        }
    }

    return result;
}

// Frobenius Norm
template <typename T>
Matrix<T> Matrix<T>::frobeniusNorm() const {
    T sum = T();
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) { 
            sum += data[i][j] * data[i][j];
        }
    }
    Matrix<T> result(1, 1);
    result(0, 0) = std::sqrt(sum);
    return result;
}