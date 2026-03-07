#include <iostream>
#include <sstream>
#include "simple_type_traits.h"

class PCErrorHandler
{
public:
    template <typename T>
    static void log(const T &msg, ErrorLevel level)
    {
        std::ostringstream oss;
        oss << msg; // works for any type that has operator<< defined

        std::string prefix;
        switch (level)
        {
        case ErrorLevel::Plain:
            std::cout << oss.str();
            return;
        case ErrorLevel::Info:
            prefix = "[INFO] ";
            break;
        case ErrorLevel::Warning:
            prefix = "[WARN] ";
            break;
        case ErrorLevel::Error:
            prefix = "[ERROR] ";
            break;
        case ErrorLevel::Fatal:
            prefix = "[FATAL] ";
            break;
        }

        std::cerr << prefix << oss.str() << std::endl;
    }

    // variadic log
    template <typename... Args>
    static void log(ErrorLevel level, Args &&...args)
    {
        std::ostringstream oss;
        (oss << ... << forward<Args>(args));

        std::string prefix;
        switch (level)
        {
        case ErrorLevel::Plain:
            std::cout << oss.str();
            return;
        case ErrorLevel::Info:
            prefix = "[INFO] ";
            break;
        case ErrorLevel::Warning:
            prefix = "[WARN] ";
            break;
        case ErrorLevel::Error:
            prefix = "[ERROR] ";
            break;
        case ErrorLevel::Fatal:
            prefix = "[FATAL] ";
            break;
        }

        std::cerr << prefix << oss.str() << std::endl;
    }

    template <typename... Args>
    [[noreturn]] static void error(Args &&...args)
    {
        std::ostringstream oss;
        (oss << ... << forward<Args>(args));
        throw std::runtime_error(oss.str());
    }
};
