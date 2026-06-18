// ============================================================================
// Matrix.tpp - Matrix class implementation
// ============================================================================

// Constructors
template <typename T>
Matrix<T>::Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c) {
    // data is initialized with default-constructed T() values
}

template <typename T>
Matrix<T>::Matrix(size_t r, size_t c, const T& initial) 
    : rows(r), cols(c), data(r * c, initial) {
    // All elements initialized to 'initial'
}

// Access elements using row-major indexing: index = row * cols + col
template <typename T>
T& Matrix<T>::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row * cols + col];
}

template <typename T>
const T& Matrix<T>::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[row * cols + col];
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

// Direct data access
template <typename T>
T* Matrix<T>::getData() {
    return data.data();
}

template <typename T>
const T* Matrix<T>::getData() const {
    return data.data();
}

// Matrix addition
template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] + other.data[i];
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
    for (size_t i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] - other.data[i];
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
                sum += data[i * cols + k] * other.data[k * other.cols + j];
            }
            result.data[i * result.cols + j] = sum;
        }
    }
    return result;
}

// Scalar multiplication
template <typename T>
Matrix<T> Matrix<T>::operator*(const T& scalar) const {
    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

// Transpose
template <typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[j * rows + i] = data[i * cols + j];
        }
    }
    return result;
}

// Print matrix
template <typename T>
void Matrix<T>::print() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << data[i * cols + j] << " ";
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

    size_t rowIndex, colIndex;

    for (size_t rBlockIdx = 0; rBlockIdx < row_blocks; ++rBlockIdx) {
        for (size_t cBlockIdx = 0; cBlockIdx < col_blocks; ++cBlockIdx) {
            for (size_t i = 0; i < bsize_row; ++i) {
                rowIndex = rBlockIdx * bsize_row + i; // Calculate row index
                for (size_t j = 0; j < bsize_col; ++j) {
                    colIndex = cBlockIdx * bsize_col; // Calculate column index
                    result.data[ (colIndex + j) * rows + rowIndex] = data[rowIndex * cols + colIndex + j];
                }
            }
        }
    }

    // Handle remainder rows
    if (row_remainder > 0) {
        for (size_t cBlockIdx = 0; cBlockIdx < col_blocks; ++cBlockIdx) {
            for (size_t i = 0; i < row_remainder; ++i) {
                rowIndex = row_blocks * bsize_row + i; // Calculate row index
                for (size_t j = 0; j < bsize_col; ++j) {
                    colIndex = cBlockIdx * bsize_col + j; // Calculate column index
                    result.data[colIndex * rows + rowIndex] = data[rowIndex * cols + colIndex];
                }
            }
        }
    }

    // Handle remainder columns
    if (col_remainder > 0) {
        for (size_t rBlockIdx = 0; rBlockIdx < row_blocks; ++rBlockIdx) {
            for (size_t i = 0; i < bsize_row; ++i) {
                rowIndex = rBlockIdx * bsize_row + i; // Calculate row index
                for (size_t j = 0; j < col_remainder; ++j) {
                    colIndex = col_blocks * bsize_col + j; // Calculate column index
                    result.data[colIndex * rows + rowIndex] = data[rowIndex * cols + colIndex];
                }
            }
        }
    }

    // Handle bottom-right remainder block
    if (row_remainder > 0 && col_remainder > 0) {
        for (size_t i = 0; i < row_remainder; ++i) {
            for (size_t j = 0; j < col_remainder; ++j) {
                rowIndex = row_blocks * bsize_row + i; // Calculate row index
                colIndex = col_blocks * bsize_col + j; // Calculate column index
                result.data[colIndex * rows + rowIndex] = data[rowIndex * cols + colIndex];
            }
        }
    }

    return result;
}

// Frobenius Norm
template <typename T>
Matrix<T> Matrix<T>::frobeniusNorm() const {
    T sum = T();
    for (size_t i = 0; i < rows * cols; ++i) {
        sum += data[i] * data[i];
    }
    Matrix<T> result(1, 1);
    result.data[0] = std::sqrt(sum);
    return result;
}