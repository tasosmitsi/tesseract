#ifndef ARM_DIVIDE_H
#define ARM_DIVIDE_H

#include "math/divide.h"
#include "arm/none.h"
// TODO: figure out which includes are actually needed

namespace math
{
    /**
      @ingroup groupFastMath
     */

    /**
      @defgroup divide Fixed point division

     */

    /**
      @addtogroup divide
      @{
     */

    /**
      @brief         Fixed point division
      @param[in]     numerator    Numerator
      @param[in]     denominator  Denominator
      @param[out]    quotient     Quotient value normalized between -1.0 and 1.0
      @param[out]    shift        Shift left value to get the unnormalized quotient
      @return        error status

      When dividing by 0, an error ARM_MATH_NANINF is returned. And the quotient is forced
      to the saturated negative or positive value.
     */
    // Specialization for q15_t
    // template <>
    ARM_DSP_ATTRIBUTE arm_status divide(q15_t numerator,
                                        q15_t denominator,
                                        q15_t *quotient,
                                        int16_t *shift)
    {
        int16_t sign = 0;
        q31_t temp;
        int16_t shiftForNormalizing;

        *shift = 0;

        sign = (numerator < 0) ^ (denominator < 0);

        if (denominator == 0)
        {
            if (sign)
            {
                *quotient = -32768;
            }
            else
            {
                *quotient = 32767;
            }
            return (ARM_MATH_NANINF);
        }

        arm_abs_q15(&numerator, &numerator, 1);
        arm_abs_q15(&denominator, &denominator, 1);

        temp = ((q31_t)numerator << 15) / ((q31_t)denominator);

        shiftForNormalizing = 17 - __CLZ(temp);
        if (shiftForNormalizing > 0)
        {
            *shift = shiftForNormalizing;
            temp = temp >> shiftForNormalizing;
        }

        if (sign)
        {
            temp = -temp;
        }

        *quotient = temp;

        return (ARM_MATH_SUCCESS);
    }

    /**
      @} end of divide group
     */

    /**
      @ingroup groupFastMath
     */

    /**
      @addtogroup divide
      @{
     */

    /**
      @brief         Fixed point division
      @param[in]     numerator    Numerator
      @param[in]     denominator  Denominator
      @param[out]    quotient     Quotient value normalized between -1.0 and 1.0
      @param[out]    shift        Shift left value to get the unnormalized quotient
      @return        error status

      When dividing by 0, an error ARM_MATH_NANINF is returned. And the quotient is forced
      to the saturated negative or positive value.
     */
    // Specialization for q31_t
    // template <>
    ARM_DSP_ATTRIBUTE arm_status divide(q31_t numerator,
                                        q31_t denominator,
                                        q31_t *quotient,
                                        int16_t *shift)
    {
        int16_t sign = 0;
        q63_t temp;
        int16_t shiftForNormalizing;

        *shift = 0;

        sign = (numerator < 0) ^ (denominator < 0);

        if (denominator == 0)
        {
            if (sign)
            {
                *quotient = 0x80000000;
            }
            else
            {
                *quotient = 0x7FFFFFFF;
            }
            return (ARM_MATH_NANINF);
        }

        arm_abs_q31(&numerator, &numerator, 1);
        arm_abs_q31(&denominator, &denominator, 1);

        temp = ((q63_t)numerator << 31) / ((q63_t)denominator);

        shiftForNormalizing = 32 - __CLZ(temp >> 31);
        if (shiftForNormalizing > 0)
        {
            *shift = shiftForNormalizing;
            temp = temp >> shiftForNormalizing;
        }

        if (sign)
        {
            temp = -temp;
        }

        *quotient = (q31_t)temp;

        return (ARM_MATH_SUCCESS);
    }

    /**
      @} end of divide group
     */
}

#endif // ARM_DIVIDE_H