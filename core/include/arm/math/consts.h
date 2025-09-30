#ifndef CONSTS_H
#define CONSTS_H

#include "arm/config.h"
#include "arm/arm_math_types.h"
#include "arm/arm_math_types_f16.h"

// Constants
#define PI 3.14159265358979f
#define PI_F64 3.14159265358979323846

// ---------------------------------------------------------------------
#ifdef ARM_FLOAT16_SUPPORTED
#ifndef float16_t
typedef float float16_t; // emulate on host
#endif
static constexpr float16_t PIF16 = 3.1415926f;
static constexpr float16_t PI16HALF = 1.5707963f;
static constexpr float16_t ATANHALFF16 = 0.463648f;
static constexpr int ATAN2_NB_COEFS_F16 = 5;
static constexpr float16_t atan2_coefs_f16[ATAN2_NB_COEFS_F16] = {
    0.0f,
    1.0f,
    0.0f,
    -0.367f,
    0.152f};
#endif // ARM_FLOAT16_SUPPORTED
// ---------------------------------------------------------------------

static constexpr _Float32 PIF32 = 3.1415926535897932384626f;
static constexpr _Float32 PIHALFF32 = 1.5707963267948966192313f;
static constexpr _Float32 ATANHALFF32 = 0.463648f;
static constexpr int ATAN2_NB_COEFS_F32 = 10;
static constexpr float32_t atan2_coefs_f32[ATAN2_NB_COEFS_F32] = {
    0.0f,
    1.0000001638308195518f,
    -0.0000228941363602264f,
    -0.3328086544578890873f,
    -0.004404814619311061f,
    0.2162217461808173258f,
    -0.0207504842057097504f,
    -0.1745263362250363339f,
    0.1340557235283553386f,
    -0.0323664125927477625f};

#define PIQ13 0x6488      // Q2.13 representation of PI
#define PIHALFQ13 0x3244  // Q2.13 representation of PI/2
#define ATANHALFQ13 0xed6 // Q2.13 representation of atan(0.5)
static constexpr int ATAN2_NB_COEFS_Q15 = 10;
static constexpr q15_t atan2_coefs_q15[ATAN2_NB_COEFS_Q15] = {
    0,      // 0x0000
    32767,  // 0x7fff
    -1,     // 0xffff
    -10905, // 0xd567
    -144,   // 0xff70
    7085,   // 0x1bad
    -680,   // 0xfd58
    -5719,  // 0xe9a9
    4393,   // 0x1129
    -1061}; // 0xfbdb

#define PIQ29 0x6487ed51       // Q2.29 representation of PI
#define PIHALF_Q29 0x3243f6a9  // Q2.29 representation of PI/2
#define ATANHALF_Q29 0xed63383 // Q2.29 representation of atan(0.5)
static constexpr int ATAN2_NB_COEFS_Q31 = 13;
static constexpr q31_t atan2_coefs_q31[ATAN2_NB_COEFS_Q31] = {
    (q31_t)0x00000000,
    (q31_t)0x7ffffffe,
    (q31_t)0x000001b6,
    (q31_t)0xd555158e,
    (q31_t)0x00036463,
    (q31_t)0x1985f617,
    (q31_t)0x001992ae,
    (q31_t)0xeed53a7f,
    (q31_t)0xf8f15245,
    (q31_t)0x2215a3a4,
    (q31_t)0xe0fab004,
    (q31_t)0x0cdd4825,
    (q31_t)0xfddbc054};

#endif // CONSTS_H
