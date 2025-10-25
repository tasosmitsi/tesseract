#ifndef UTILIS_H
#define UTILIS_H

#include <string>
#include <vector>
#include <iostream>
#include <cxxabi.h>
#include <iomanip>
#include <algorithm>
#include <memory>

#include "../core/include/config.h"

void tick();

uint tock(std::string message);

uint tock();

template <typename T>
void print_expr_type(const T &)
{
    int status;
    std::unique_ptr<char[], void (*)(void *)> demangled(
        abi::__cxa_demangle(typeid(T).name(), 0, 0, &status), std::free);
    std::cout << "Expression type: " << (status == 0 ? demangled.get() : typeid(T).name()) << std::endl;
}

std::string demangleTypeName(const std::type_info &ti);

std::string executePythonAndGetString(const std::string &python_code);

void removeNewlines(std::string &str);

template <typename MatrixType>
std::string toNumpyArray(const MatrixType &matrix)
{
    my_size_t rows = matrix.getDim(0);
    my_size_t cols = matrix.getDim(1);
    std::ostringstream oss;
    oss << "[";

    for (my_size_t i = 0; i < rows; ++i)
    {
        oss << "[";
        for (my_size_t j = 0; j < cols; ++j)
        {
            // Print each element with 3 decimal places
            oss << std::fixed << std::setprecision(3) << matrix(i, j);
            if (j < cols - 1)
            {
                oss << ", "; // Comma between elements in a row
            }
        }
        oss << "]";
        if (i < rows - 1)
        {
            oss << ", "; // Comma between rows
        }
    }

    oss << "]";
    return oss.str();
}

template <typename MatrixType>
std::string toFormattedNumpyArray(const MatrixType &matrix)
{
    my_size_t rows = matrix.getDim(0);
    my_size_t cols = matrix.getDim(1);
    std::ostringstream oss;
    oss << "[";

    for (my_size_t i = 0; i < rows; ++i)
    {
        oss << "[";
        for (my_size_t j = 0; j < cols; ++j)
        {
            // Print each element with 3 decimal places
            oss << std::fixed << std::setprecision(3) << matrix(i, j);
            if (j < cols - 1)
            {
                oss << " "; // Space between elements in a row
            }
        }
        oss << "]";
        if (i < rows - 1)
        {
            oss << " "; // Space between rows
        }
    }

    oss << "]";
    return oss.str();
}

std::vector<std::string> splitStringByComma(const std::string &input);

#endif
