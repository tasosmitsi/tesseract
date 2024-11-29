// give me a func that prints hello
#ifndef UTILIS_H
#define UTILIS_H

#include <Python.h>
#include <iomanip>

std::string executePythonAndGetString(const std::string &python_code)
{
    std::string result;

    // Initialize Python
    if (!Py_IsInitialized())
    {
        Py_Initialize();
    }

    if (!Py_IsInitialized())
    {
        throw std::runtime_error("Failed to initialize Python interpreter.");
    }

    try
    {
        // Run the Python code
        PyObject *globals = PyDict_New(); // Create a new dictionary for global variables
        PyObject *locals = PyDict_New();  // Local variables (can be shared with globals)
        if (PyRun_String(python_code.c_str(), Py_file_input, globals, locals) == nullptr)
        {
            PyErr_Print(); // Print any Python errors
            throw std::runtime_error("Error executing Python code.");
        }

        // Retrieve the output_string variable from Python
        PyObject *pyOutputString = PyDict_GetItemString(locals, "output_string");
        if (pyOutputString && PyUnicode_Check(pyOutputString))
        {
            result = PyUnicode_AsUTF8(pyOutputString);
        }
        else
        {
            throw std::runtime_error("Failed to retrieve Python output string.");
        }

        // Clean up
        Py_DECREF(globals);
        Py_DECREF(locals);
    }
    catch (const std::exception &e)
    {
        Py_Finalize(); // Ensure Python is finalized on error
        throw;
    }

    // Finalize the Python interpreter
    Py_Finalize();

    return result;
}

void removeNewlines(std::string &str)
{
    // Remove all '\n' (newline) and '\r' (carriage return) characters
    str.erase(std::remove_if(str.begin(), str.end(), [](char ch)
                             { return ch == '\n' || ch == '\r'; }),
              str.end());
}

template <class T, long unsigned int Rows, long unsigned int Cols>
std::string toNumpyArray(const Matrix<T, Rows, Cols> &mat)
{
    my_size_t rows = mat.getDim(0);
    my_size_t cols = mat.getDim(1);
    std::ostringstream oss;
    oss << "[";

    for (my_size_t i = 0; i < rows; ++i)
    {
        oss << "[";
        for (my_size_t j = 0; j < cols; ++j)
        {
            // Print each element with 3 decimal places
            oss << std::fixed << std::setprecision(3) << mat(i, j);
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

template <class T, long unsigned int Rows, long unsigned int Cols>
std::string toFormattedNumpyArray(const Matrix<T, Rows, Cols> &mat)
{
    my_size_t rows = mat.getDim(0);
    my_size_t cols = mat.getDim(1);
    std::ostringstream oss;
    oss << "[";

    for (my_size_t i = 0; i < rows; ++i)
    {
        oss << "[";
        for (my_size_t j = 0; j < cols; ++j)
        {
            // Print each element with 3 decimal places
            oss << std::fixed << std::setprecision(3) << mat(i, j);
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

std::vector<std::string> splitStringByComma(const std::string &input)
{
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string token;

    while (std::getline(ss, token, ','))
    {
        result.push_back(token);
    }

    return result;
}

#endif