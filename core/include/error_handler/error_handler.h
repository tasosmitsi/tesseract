#pragma once
#include <string>

enum class ErrorLevel
{
    Plain,
    Info,
    Warning,
    Error,
    Fatal
};

template <typename Impl>
class ErrorHandler
{
public:
    template <typename T>
    static void log(const T &msg, ErrorLevel level = ErrorLevel::Plain)
    {
        Impl::log(msg, level);
    }

    template <typename T>
    static void error(const T &msg)
    {
        Impl::error(msg);
    }
};
