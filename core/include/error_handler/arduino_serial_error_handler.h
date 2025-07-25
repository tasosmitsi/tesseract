#pragma once
#include <Arduino.h>

class SerialErrorHandler
{
public:
    static void log(const std::string &msg, ErrorLevel level)
    {
        Serial.print("[ERR] ");
        Serial.println(msg.c_str());
    }

    static void error(const std::string &msg)
    {
        log(msg, ErrorLevel::Fatal);
        while (true)
            ; // halt forever
    }
};
