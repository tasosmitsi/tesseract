#ifndef ATAN_H
#define ATAN_H

namespace math
{
    template <typename T>
    T atan(T x)
    {
        std::cout << "arctangent generic" << std::endl;
        return x; // Placeholder implementation
    }

    template <typename T>
    T atan2(T x)
    {
        std::cout << "arctangent 2 generic" << std::endl;
        return x; // Placeholder implementation
    }
}

#endif // ATAN_H