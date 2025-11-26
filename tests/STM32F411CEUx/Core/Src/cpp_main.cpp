#include "cpp_main.hpp"
extern "C"
{
#include "main.h" // HAL headers
}

#include <array> // <-- C++ library example
#include <algorithm>
#include <cstdio>

extern "C" void cpp_main(void)
{
    // Create a fixed-size array of 3 integers

    std::array<int, 3> nums = {1, 2, 3};

    // Compute sum
    int sum = 0;
    for (auto n : nums)
        sum += n;

    // Infinite loop
    while (1)
    {
        // Example: toggle LED (if configured)
        // HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
        HAL_Delay(500);
        printf("Cpp is running... \r\n");
        printf("Sum is: %d \r\n", sum);

    }
}