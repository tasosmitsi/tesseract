#pragma once
#include "config.h"

template <typename Expr1, typename Expr2>
void checkDimsMatch(const Expr1 &lhs, const Expr2 &rhs, const char *opName) TESSERACT_CONDITIONAL_NOEXCEPT
{
#ifdef RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH
    if (lhs.getNumDims() != rhs.getNumDims()) [[unlikely]]
        MyErrorHandler::error(opName, ": dimension count mismatch");
#endif

#ifdef RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH
    for (my_size_t i = 0; i < lhs.getNumDims(); ++i)
    {
        if (lhs.getDim(i) != rhs.getDim(i)) [[unlikely]]
            MyErrorHandler::error(opName, ": dimension size mismatch at dimension ", i);
    }
#endif
}