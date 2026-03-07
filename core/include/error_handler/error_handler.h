#pragma once
#include "simple_type_traits.h"

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

    template <typename... Args>
    static void log(ErrorLevel level, Args &&...args)
    {
        Impl::log(level, forward<Args>(args)...);
    }

    template <typename T>
    [[noreturn]] static void error(const T &msg)
    {
        Impl::error(msg);
    }

    template <typename... Args>
        requires(sizeof...(Args) > 1)
    [[noreturn]] static void error(Args &&...args)
    {
        Impl::error(forward<Args>(args)...);
    }
};
