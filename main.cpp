// ============================================================================
// main.cpp - Example usage
// ============================================================================
#include "Matrix.hpp"

int main() {
    // Create a 3x3 matrix of integers
    Matrix<int> m1(3, 3);
    
    // Fill with values
    int counter = 1;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m1(i, j) = counter++;
        }
    }

    std::cout << "Matrix m1:" << std::endl;
    m1.print();

    // Create another matrix
    Matrix<int> m2(3, 3, 2);
    std::cout << "\nMatrix m2 (all 2s):" << std::endl;
    m2.print();

    // Matrix addition
    Matrix<int> m3 = m1 + m2;
    std::cout << "\nm1 + m2:" << std::endl;
    m3.print();

    // Scalar multiplication
    Matrix<int> m4 = m1 * 3;
    std::cout << "\nm1 * 3:" << std::endl;
    m4.print();

    // Transpose
    Matrix<int> m5 = m1.transpose();
    std::cout << "\nTranspose of m1:" << std::endl;
    m5.print();

    // Matrix multiplication
    Matrix<double> a(2, 3);
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;

    Matrix<double> b(3, 2);
    b(0, 0) = 7; b(0, 1) = 8;
    b(1, 0) = 9; b(1, 1) = 10;
    b(2, 0) = 11; b(2, 1) = 12;

    std::cout << "\nMatrix a (2x3):" << std::endl;
    a.print();
    std::cout << "\nMatrix b (3x2):" << std::endl;
    b.print();

    Matrix<double> c = a * b;
    std::cout << "\na * b:" << std::endl;
    c.print();

    // Run optimized transpose
    Matrix<int> m6 = m1.optTranspose(2, 2);
    std::cout << "\nm6 Optimized Transpose of m1:" << std::endl;
    m6.print();

    // Verify correctness by comparing with standard transpose
    Matrix<int> mdiff = m5 - m6;
    std::cout << "\nDifference between original (m5) and optimized transpose (m6):" << std::endl;
    mdiff.print();
    std::cout << "\nFrobenius Norm of mdiff:" << std::endl;
    mdiff.frobeniusNorm().print();
    return 0;
}