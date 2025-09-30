#ifndef ARM_ATAN_H
#define ARM_ATAN_H

#include "arm/config.h"
#include "arm/math/consts.h"
#include "arm/none.h"
#include "arm/utils.h" // TODO: check if needed
#include "arm/math/arm_devide.h"
#include "arm/arm_math_types.h"
// TODO: figure out which includes are actually needed

#ifdef ARM_FLOAT16_SUPPORTED

__STATIC_FORCEINLINE float16_t arm_atan_limited_f16(float16_t x)
{
    float16_t res = atan2_coefs_f16[ATAN2_NB_COEFS_F16 - 1];
    int i = 1;
    for (i = 1; i < ATAN2_NB_COEFS_F16; i++)
    {
        res = (_Float16)x * (_Float16)res + (_Float16)atan2_coefs_f16[ATAN2_NB_COEFS_F16 - 1 - i];
    }

    return (res);
}

__STATIC_FORCEINLINE float16_t arm_atan_f16(float16_t x)
{
    int sign = 0;
    _Float16 res = 0.0f16;

    if ((_Float16)x < 0.0f16)
    {
        sign = 1;
        x = -(_Float16)x;
    }

    if ((_Float16)x > 1.0f16)
    {
        x = 1.0f16 / (_Float16)x;
        res = (_Float16)PI16HALF - (_Float16)arm_atan_limited_f16(x);
    }
    else
    {
        res += (_Float16)arm_atan_limited_f16(x);
    }

    if (sign)
    {
        res = -(_Float16)res;
    }

    return ((float16_t)res);
}

#endif // ARM_FLOAT16_SUPPORTED

__STATIC_FORCEINLINE float32_t arm_atan_limited_f32(float32_t x)
{
    float32_t res = atan2_coefs_f32[ATAN2_NB_COEFS_F32 - 1];
    int i = 1;
    for (i = 1; i < ATAN2_NB_COEFS_F32; i++)
    {
        res = x * res + atan2_coefs_f32[ATAN2_NB_COEFS_F32 - 1 - i];
    }

    return (res);
}

__STATIC_FORCEINLINE float32_t arm_atan_f32(float32_t x)
{
    int sign = 0;
    float32_t res = 0.0f;

    if (x < 0.0f)
    {
        sign = 1;
        x = -x;
    }

    if (x > 1.0f)
    {
        x = 1.0f / x;
        res = PIHALFF32 - arm_atan_limited_f32(x);
    }
    else
    {
        res += arm_atan_limited_f32(x);
    }

    if (sign)
    {
        res = -res;
    }

    return (res);
}

__STATIC_FORCEINLINE q15_t arm_atan_limited_q15(q15_t x)
{
    q31_t res = (q31_t)atan2_coefs_q15[ATAN2_NB_COEFS_Q15 - 1];
    int i = 1;
    for (i = 1; i < ATAN2_NB_COEFS_Q15; i++)
    {
        res = ((q31_t)x * res) >> 15U;
        res = res + ((q31_t)atan2_coefs_q15[ATAN2_NB_COEFS_Q15 - 1 - i]);
    }

    res = __SSAT(res >> 2, 16);

    return (res);
}

__STATIC_FORCEINLINE q15_t arm_atan_q15(q15_t y, q15_t x)
{
    int sign = 0;
    q15_t res = 0;

    if (y < 0)
    {
        /* Negate y */
#if defined(ARM_MATH_DSP)
        y = __QSUB16(0, y);
#else
        y = (y == (q15_t)0x8000) ? (q15_t)0x7fff : -y;
#endif

        sign = 1 - sign;
    }

    if (x < 0)
    {
        sign = 1 - sign;

        /* Negate x */
#if defined(ARM_MATH_DSP)
        x = __QSUB16(0, x);
#else
        x = (x == (q15_t)0x8000) ? (q15_t)0x7fff : -x;
#endif
    }

    if (y > x)
    {
        q15_t ratio;
        int16_t shift;

        math::divide(x, y, &ratio, &shift);

        /* Shift ratio by shift */
        if (shift >= 0)
        {
            ratio = __SSAT(((q31_t)ratio << shift), 16);
        }
        else
        {
            ratio = (ratio >> -shift);
        }

        res = PIHALFQ13 - arm_atan_limited_q15(ratio);
    }
    else
    {
        q15_t ratio;
        int16_t shift;

        math::divide(y, x, &ratio, &shift);

        /* Shift ratio by shift */
        if (shift >= 0)
        {
            ratio = __SSAT(((q31_t)ratio << shift), 16);
        }
        else
        {
            ratio = (ratio >> -shift);
        }

        res = arm_atan_limited_q15(ratio);
    }

    if (sign)
    {
        /* Negate res */
#if defined(ARM_MATH_DSP)
        res = __QSUB16(0, res);
#else
        res = (res == (q15_t)0x8000) ? (q15_t)0x7fff : -res;
#endif
    }

    return (res);
}

__STATIC_FORCEINLINE q31_t arm_atan_limited_q31(q31_t x)
{
    q63_t res = (q63_t)atan2_coefs_q31[ATAN2_NB_COEFS_Q31 - 1];
    int i = 1;
    for (i = 1; i < ATAN2_NB_COEFS_Q31; i++)
    {
        res = ((q63_t)x * res) >> 31U;
        res = res + ((q63_t)atan2_coefs_q31[ATAN2_NB_COEFS_Q31 - 1 - i]);
    }

    return (clip_q63_to_q31(res >> 2));
}

__STATIC_FORCEINLINE q31_t arm_atan_q31(q31_t y, q31_t x)
{
    int sign = 0;
    q31_t res = 0;

    if (y < 0)
    {
        /* Negate y */
#if defined(ARM_MATH_DSP)
        y = __QSUB(0, y);
#else
        y = (y == INT32_MIN) ? INT32_MAX : -y;
#endif

        sign = 1 - sign;
    }

    if (x < 0)
    {
        sign = 1 - sign;

        /* Negate x */
#if defined(ARM_MATH_DSP)
        x = __QSUB(0, x);
#else
        x = (x == INT32_MIN) ? INT32_MAX : -x;
#endif
    }

    if (y > x)
    {
        q31_t ratio;
        int16_t shift;

        math::divide(x, y, &ratio, &shift);

        /* Shift ratio by shift */
        if (shift >= 0)
        {
            ratio = clip_q63_to_q31((q63_t)ratio << shift);
        }
        else
        {
            ratio = (ratio >> -shift);
        }

        res = PIHALF_Q29 - arm_atan_limited_q31(ratio);
    }
    else
    {
        q31_t ratio;
        int16_t shift;

        math::divide(y, x, &ratio, &shift);

        /* Shift ratio by shift */
        if (shift >= 0)
        {
            ratio = clip_q63_to_q31((q63_t)ratio << shift);
        }
        else
        {
            ratio = (ratio >> -shift);
        }

        res = arm_atan_limited_q31(ratio);
    }

    if (sign)
    {
        /* Negate res */
#if defined(ARM_MATH_DSP)
        res = __QSUB(0, res);
#else
        res = (res == INT32_MIN) ? INT32_MAX : -res;
#endif
    }

    return (res);
}

namespace math
{
#ifdef ARM_FLOAT16_SUPPORTED

    /**
    @brief       Arc Tangent of y/x using sign of y and x to get right quadrant
    @param[in]   y  y coordinate
    @param[in]   x  x coordinate
    @param[out]  result  Result
    @return  error status.

    @par         Compute the Arc tangent of y/x:
                    The sign of y and x are used to determine the right quadrant
                    and compute the right angle. Returned value is between -Pi and Pi.

    */
    ARM_DSP_ATTRIBUTE arm_status atan2<float16_t>(float16_t y, float16_t x, float16_t *result)
    {
        if ((_Float16)x > 0.0f16)
        {
            *result = ::arm_atan_f16((_Float16)y / (_Float16)x);
            return (ARM_MATH_SUCCESS);
        }
        if ((_Float16)x < 0.0f16)
        {
            if ((_Float16)y > 0.0f16)
            {
                *result = (_Float16)::arm_atan_f16((_Float16)y / (_Float16)x) + (_Float16)PIF16;
            }
            else if ((_Float16)y < 0.0f16)
            {
                *result = (_Float16)::arm_atan_f16((_Float16)y / (_Float16)x) - (_Float16)PIF16;
            }
            else
            {
                if (signbit((float)y))
                {
                    *result = -(_Float16)PIF16;
                }
                else
                {
                    *result = PIF16;
                }
            }
            return (ARM_MATH_SUCCESS);
        }
        if ((_Float16)x == 0.0f16)
        {
            if ((_Float16)y > 0.0f16)
            {
                *result = PI16HALF;
                return (ARM_MATH_SUCCESS);
            }
            if ((_Float16)y < 0.0f16)
            {
                *result = -(_Float16)PI16HALF;
                return (ARM_MATH_SUCCESS);
            }
        }

        return (ARM_MATH_NANINF);
    }

#endif // ARM_FLOAT16_SUPPORTED

    /**
      @ingroup groupFastMath
     */

    /**
      @defgroup atan2 ArcTan2

      Computing Arc tangent only using the ratio y/x is not enough to determine the angle
      since there is an indeterminacy. Opposite quadrants are giving the same ratio.

      ArcTan2 is not using y/x to compute the angle but y and x and use the sign of y and x
      to determine the quadrant.

     */

    /**
      @addtogroup atan2
      @{
     */

    /**
      @brief       Arc Tangent of y/x using sign of y and x to get right quadrant
      @param[in]   y  y coordinate
      @param[in]   x  x coordinate
      @param[out]  result  Result
      @return  error status.

      @par         Compute the Arc tangent of y/x:
                       The sign of y and x are used to determine the right quadrant
                       and compute the right angle. Returned value is between -Pi and Pi.
    */
    ARM_DSP_ATTRIBUTE arm_status atan2(float32_t y, float32_t x, float32_t *result)
    {
        if (x > 0.0f)
        {
            *result = ::arm_atan_f32(y / x);
            return (ARM_MATH_SUCCESS);
        }
        if (x < 0.0f)
        {
            if (y > 0.0f)
            {
                *result = ::arm_atan_f32(y / x) + PI;
            }
            else if (y < 0.0f)
            {
                *result = ::arm_atan_f32(y / x) - PI;
            }
            else
            {
                if (signbit(y))
                {
                    *result = -PI;
                }
                else
                {
                    *result = PI;
                }
            }
            return (ARM_MATH_SUCCESS);
        }
        if (x == 0.0f)
        {
            if (y > 0.0f)
            {
                *result = PIHALFF32;
                return (ARM_MATH_SUCCESS);
            }
            if (y < 0.0f)
            {
                *result = -PIHALFF32;
                return (ARM_MATH_SUCCESS);
            }
        }

        return (ARM_MATH_NANINF);
    }

    /**
      @} end of atan2 group
     */

    /**
      @ingroup groupFastMath
     */

    /**
      @addtogroup atan2
      @{
     */

    /**
      @brief       Arc Tangent of y/x using sign of y and x to get right quadrant
      @param[in]   y  y coordinate
      @param[in]   x  x coordinate
      @param[out]  result  Result in Q2.13
      @return  error status.

      @par         Compute the Arc tangent of y/x:
                       The sign of y and x are used to determine the right quadrant
                       and compute the right angle. Returned value is between -Pi and Pi.
    */
    ARM_DSP_ATTRIBUTE arm_status atan2(q15_t y, q15_t x, q15_t *result)
    {
        if (x > 0)
        {
            *result = ::arm_atan_q15(y, x);
            return (ARM_MATH_SUCCESS);
        }
        if (x < 0)
        {
            if (y > 0)
            {
                *result = ::arm_atan_q15(y, x) + PIQ13;
            }
            else if (y < 0)
            {
                *result = ::arm_atan_q15(y, x) - PIQ13;
            }
            else
            {
                *result = PIQ13;
            }
            return (ARM_MATH_SUCCESS);
        }
        if (x == 0)
        {
            if (y > 0)
            {
                *result = PIHALFQ13;
                return (ARM_MATH_SUCCESS);
            }
            if (y < 0)
            {
                *result = -PIHALFQ13;
                return (ARM_MATH_SUCCESS);
            }
        }

        return (ARM_MATH_NANINF);
    }

    /**
      @} end of atan2 group
     */

    /**
      @ingroup groupFastMath
     */

    /**
      @addtogroup atan2
      @{
     */

    /**
      @brief       Arc Tangent of y/x using sign of y and x to get right quadrant
      @param[in]   y  y coordinate
      @param[in]   x  x coordinate
      @param[out]  result  Result in Q2.29
      @return  error status.

      @par         Compute the Arc tangent of y/x:
                       The sign of y and x are used to determine the right quadrant
                       and compute the right angle. Returned value is between -Pi and Pi.
    */
    ARM_DSP_ATTRIBUTE arm_status atan2(q31_t y, q31_t x, q31_t *result)
    {
        if (x > 0)
        {
            *result = ::arm_atan_q31(y, x);
            return (ARM_MATH_SUCCESS);
        }
        if (x < 0)
        {
            if (y > 0)
            {
                *result = ::arm_atan_q31(y, x) + PIQ29;
            }
            else if (y < 0)
            {
                *result = ::arm_atan_q31(y, x) - PIQ29;
            }
            else
            {
                *result = PIQ29;
            }
            return (ARM_MATH_SUCCESS);
        }
        if (x == 0)
        {
            if (y > 0)
            {
                *result = PIHALF_Q29;
                return (ARM_MATH_SUCCESS);
            }
            if (y < 0)
            {
                *result = -PIHALF_Q29;
                return (ARM_MATH_SUCCESS);
            }
        }

        return (ARM_MATH_NANINF);
    }

    /**
      @} end of atan2 group
     */

}

#endif // ARM_ATAN_H