#include "utilities.h"

#include <Python.h>
#include <chrono>
#include <iostream>

using namespace std::chrono;

auto start = high_resolution_clock::now();
void tick()
{
    start = high_resolution_clock::now();
}
uint tock(std::string message)
{
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << message << ": "
              << duration.count() << " microseconds" << std::endl;

    return duration.count();
}

uint tock()
{
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken: "
              << duration.count() << " microseconds" << std::endl;

    return duration.count();
}

std::string demangleTypeName(const std::type_info &ti)
{
    int status = 0;
    std::unique_ptr<char, void (*)(void *)> demangled(
        abi::__cxa_demangle(ti.name(), nullptr, nullptr, &status),
        std::free);

    return (status == 0) ? demangled.get() : ti.name();
}

std::string executePythonAndGetString(const std::string &python_code)
{
    std::string result;

    // Initialize Python only once
    static bool isPythonInitialized = false;
    if (!isPythonInitialized)
    {
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            throw std::runtime_error("Failed to initialize Python interpreter.");
        }
        isPythonInitialized = true;
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
        PyErr_Print();
        throw; // Re-throw the exception
    }

    return result;
}

void removeNewlines(std::string &str)
{
    // Remove all '\n' (newline) and '\r' (carriage return) characters
    str.erase(std::remove_if(str.begin(), str.end(), [](char ch)
                             { return ch == '\n' || ch == '\r'; }),
              str.end());
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
