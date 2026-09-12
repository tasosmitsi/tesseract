#include <catch_amalgamated.hpp>
#include "fused/fused_tensor.h"
#include "fused/fused_matrix.h"
#include "algebra/algebraic_traits.h"
#include "utilities.h"
#include "utilities/expr_diag.h"
#include "utilities/cycle_counter/cycle_counter.h"

TEMPLATE_TEST_CASE("Min/Max", "[min_max]", double, float, int32_t, int64_t)
{
    using T = TestType;
    CycleCounter cc;

    SECTION("FusedTensorND")
    {
        // FusedTensorND<T, 6, 5> fmat1;
        // FusedTensorND<T, 6, 5> test;
        // FusedTensorND<T, 5, 6> fmat2;
        FusedTensorND<T, 100, 99> result, A, B, D;
        FusedTensorND<T, 99, 100> C;

        A.setHomogen(2.0);
        B.setHomogen(2.0);
        C.setHomogen(3.0);
        D.setHomogen(3.0);

        for (int i = 0; i < 1000; ++i)
        {
            cc.start();
            result = A * B + D + A * B - D - D + A * B * (T)2.0 - D * (T)3.1 + (T)4.145 * A - (T)8.118 * D;
            cc.stop();
        }
        std::cout << "Time for expression evaluation: " << cc.avg_cycles() << " cycles" << std::endl;
        cc.reset();

        auto lol = A * B + D + A * B - D - D + A * B * (T)2.0 - D * (T)2.0 + (T)2.0 * A - (T)2.0 * D;
        expr_diag::print_expr<decltype(lol)>();

        // result.printND(true);

        // result = C + A * B;

        // result = -(A * B) + C;

        // result = -(A * (T)2.0) + C;

        // result = -((T)2.0 * B) + C;

        // result = A * (T)2.0 + C;

        // result = (T)2.0 * A + C;

        // result = C + A * (T)2.0;

        // result = C + (T)2.0 * A;

        // result = A * B - C;

        // result = C - A * B;

        // // -(A * B) - C → Fnms (via negated BinaryExpr)
        // result = -(A * B) - C;

        // // -(A * scalar) - C → Fnms
        // result = -(A * (T)2.0) - C;

        // result = -((T)2.0 * B) - C;

        // // (A * scalar) - C → Fms
        // result = A * (T)2.0 - C;

        // result = (T)2.0 * A - C;

        // // C - (A * scalar) → Fnma
        // result = C - A * (T)2.0;

        // result = C - (T)2.0 * A;

        // T minVal = min(result);
        // T maxVal = max(result);
        // T sumVal = sum(result);

        // std::cout << "Min: " << minVal << std::endl;
        // std::cout << "Max: " << maxVal << std::endl;
        // std::cout << "Sum: " << sumVal << std::endl;

        // FusedTensorND<T, 6, 5> fmat1;
        // FusedTensorND<T, 5, 6> fmat2;

        // // Fill with unique values per element
        // for (int i = 0; i < 6; i++)
        //     for (int j = 0; j < 5; j++)
        //         fmat1(i, j) = T(i * 5 + j + 1); // 1..30

        // for (int i = 0; i < 5; i++)
        //     for (int j = 0; j < 6; j++)
        //         fmat2(i, j) = T(0);

        // fmat1.printND(true);
        // fmat2.printND(true);

        // T sumVal = sum(fmat1 + fmat2.transpose_view());
        // // Should be 1+2+...+30 = 465
        // // If bug exists, will read padding zeros and get a different number

        // // std::cout << "Min: " << minVal << std::endl;
        // // // std::cout << "Max: " << maxVal << std::endl;
        // std::cout << "Sum: " << sumVal << std::endl;

        // result = fmat1 + fmat2.transpose_view();

        // result.printND(true);

        // result.print_flat_data();
        // result.print_access_policy_info();
        // std::cout << "--------" << std::endl;
        // fmat2.print();

        // result = min(max(fmat1.transpose_view(), (T)5.0), (T)10.0);

        // auto expr = sum(fmat1 + fmat2.transpose_view());

        // result = fmat1.transpose_view() + fmat2;

        // fmat1.transpose_view() + (T)1.0 == fmat2 + fmat2;

        // // std::cout << "Result:" << std::endl;

        // result = fmat1.transpose_view();
        // // result.print();
        // // result.printLayoutInfo();
        // fmat1.printLayoutInfo();
        // fmat1.transpose_view().printLayoutInfo();

        // std::cout << "--------" << std::endl;
        // std::cout << "the result tensor:" << std::endl;
        // result.printLayoutInfo();

        // // std::cout << "--------" << std::endl;
        // // result = result + fmat2;

        // // result.print();

        // std::cout << "--------" << std::endl;

        // std::cout << expr << std::endl;

        // expr_diag::print_expr<decltype(expr)>();
    }
}