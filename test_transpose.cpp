#include "Matrix.hpp"
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>

/**
clang++ -std=c++17 test_transpose.cpp -o test_transpose
./test_transpose
 */

// Test configuration
template <typename T>
struct TestConfig {
    size_t rows;
    size_t cols;
    T min_value;
    T max_value;
    size_t block_size_row;
    size_t block_size_col;
};

// Generate a random matrix
template <typename T>
Matrix<T> generateRandomMatrix(size_t rows, size_t cols, T min_val, T max_val) {
    Matrix<T> m(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dis(min_val, max_val);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                m(i, j) = dis(gen);
            }
        }
    } else {
        std::uniform_real_distribution<T> dis(min_val, max_val);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                m(i, j) = dis(gen);
            }
        }
    }
    
    return m;
}

// Calculate maximum absolute difference between two matrices
template <typename T>
T maxAbsDifference(const Matrix<T>& m1, const Matrix<T>& m2) {
    if (m1.getRows() != m2.getRows() || m1.getCols() != m2.getCols()) {
        throw std::invalid_argument("Matrices must have same dimensions");
    }
    
    T max_diff = T();
    for (size_t i = 0; i < m1.getRows(); ++i) {
        for (size_t j = 0; j < m1.getCols(); ++j) {
            T diff = std::abs(m1(i, j) - m2(i, j));
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}

// Check if two matrices are equal within tolerance
template <typename T>
bool matricesEqual(const Matrix<T>& m1, const Matrix<T>& m2, T tolerance = T()) {
    if (m1.getRows() != m2.getRows() || m1.getCols() != m2.getCols()) {
        return false;
    }
    
    for (size_t i = 0; i < m1.getRows(); ++i) {
        for (size_t j = 0; j < m1.getCols(); ++j) {
            if (std::abs(m1(i, j) - m2(i, j)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

// Run a single test
template <typename T>
bool runTest(const TestConfig<T>& config, bool verbose = false) {
    std::cout << "Testing " << config.rows << "x" << config.cols 
              << " matrix with type " << typeid(T).name() 
              << " (block size: " << config.block_size_row << "x" << config.block_size_col << ")"
              << std::endl;
    
    // Generate random matrix
    Matrix<T> original = generateRandomMatrix<T>(config.rows, config.cols, 
                                                   config.min_value, config.max_value);
    
    // Time standard transpose
    auto start1 = std::chrono::high_resolution_clock::now();
    Matrix<T> standard_result = original.transpose();
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    
    // Time optimized transpose
    auto start2 = std::chrono::high_resolution_clock::now();
    Matrix<T> optimized_result = original.optTranspose(config.block_size_row, config.block_size_col);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    
    // Calculate difference
    Matrix<T> diff = standard_result - optimized_result;
    T max_diff = maxAbsDifference(standard_result, optimized_result);
    
    // Set tolerance based on type
    T tolerance = std::is_integral<T>::value ? T() : T(1e-6);
    bool passed = matricesEqual(standard_result, optimized_result, tolerance);
    
    // Print results
    std::cout << "  Standard transpose time: " << std::setw(8) << duration1.count() << " μs" << std::endl;
    std::cout << "  Optimized transpose time: " << std::setw(8) << duration2.count() << " μs" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
              << (double)duration1.count() / duration2.count() << "x" << std::endl;
    std::cout << "  Max absolute difference: " << max_diff << std::endl;
    std::cout << "  Test " << (passed ? "PASSED ✓" : "FAILED ✗") << std::endl;
    
    if (verbose && config.rows <= 10 && config.cols <= 10) {
        std::cout << "\n  Original matrix:" << std::endl;
        original.print();
        std::cout << "\n  Standard transpose:" << std::endl;
        standard_result.print();
        std::cout << "\n  Optimized transpose:" << std::endl;
        optimized_result.print();
        if (!passed) {
            std::cout << "\n  Difference:" << std::endl;
            diff.print();
        }
    }
    
    std::cout << std::endl;
    return passed;
}

// Run test suite
void runTestSuite() {
    int passed = 0;
    int total = 0;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Matrix Transpose Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Conif
    // number of rows, number of columns, min value, max value, block size row, block size column

    // Test 1: Small integer matrix
    TestConfig<int> test1 = {5, 5, -100, 100, 2, 2};
    total++;
    if (runTest(test1, true)) passed++;
    
    // Test 2: Rectangular integer matrix
    TestConfig<int> test2 = {10, 20, -1000, 1000, 4, 4};
    total++;
    if (runTest(test2)) passed++;
    
    // Test 3: Large square integer matrix
    TestConfig<int> test3 = {1000, 1000, -10000, 10000, 16, 16};
    total++;
    if (runTest(test3)) passed++;
    
    // Test 4: Very large square integer matrix
    TestConfig<int> test4 = {1000, 1000, -10000, 10000, 32, 32};
    total++;
    if (runTest(test4)) passed++;
    
    // Test 5: Small double matrix
    TestConfig<double> test5 = {5, 5, -100.0, 100.0, 2, 2};
    total++;
    if (runTest(test5, true)) passed++;
    
    // Test 6: Large double matrix
    TestConfig<double> test6 = {5000, 5000, -1000.0, 1000.0, 64, 64};
    total++;
    if (runTest(test6)) passed++;
    
    // Test 7: Non-square double matrix (tall)
    TestConfig<double> test7 = {3000, 1000, -500.0, 500.0, 8, 8};
    total++;
    if (runTest(test7)) passed++;
    
    // Test 8: Non-square double matrix (wide)
    TestConfig<double> test8 = {1000, 3000, -500.0, 500.0, 8, 8};
    total++;
    if (runTest(test8)) passed++;
    
    // Test 9: Float matrix
    TestConfig<float> test9 = {150, 150, -100.0f, 100.0f, 8, 8};
    total++;
    if (runTest(test9)) passed++;

    // Test 10: Large Square Float matrix
    TestConfig<float> test10 = {1000, 1000, -100.0f, 100.0f, 16, 16};
    total++;
    if (runTest(test10)) passed++;
    
    // Summary
    std::cout << "========================================" << std::endl;
    std::cout << "Test Summary: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;
}

int main() {
    runTestSuite();
    return 0;
}