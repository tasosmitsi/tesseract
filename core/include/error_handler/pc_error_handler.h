#include <iostream>
#include <sstream>

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

    template <typename T>
    static void error(const T &msg)
    {
        std::ostringstream oss;
        oss << msg;

        throw std::runtime_error(oss.str());
    }
};
