#pragma once
#include <Arduino.h>
#include "simple_type_traits.h"

class SerialErrorHandler
{
public:
    static void log(const std::string &msg, ErrorLevel level)
    {
        const char *prefix;
        switch (level)
        {
        case ErrorLevel::Plain:
            prefix = "";
            break;
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
        Serial.print(prefix);
        Serial.println(msg.c_str());
    }

    template <typename... Args>
    static void log(ErrorLevel level, Args &&...args)
    {
        const char *prefix;
        switch (level)
        {
        case ErrorLevel::Plain:
            prefix = "";
            break;
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
        Serial.print(prefix);
        (Serial.print(forward<Args>(args)), ...);
        Serial.println();
    }

    template <typename... Args>
    [[noreturn]] static void error(Args &&...args)
    {
        log(ErrorLevel::Fatal, forward<Args>(args)...);
        while (true)
            ; // halt forever
    }
};