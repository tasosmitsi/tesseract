#pragma once

#include "config.h"
#include "fused/fused_tensor.h"
#include "helper_traits.h"
#include "interpolation/axis.h"

/**
 * @file lookup_table.h
 * @brief Interpolated lookup tables over scalar values.
 *
 * A lookup table holds one value per grid point and blends the surrounding ones
 * at query time. Each axis maps a query to a cell and a fraction into that cell;
 * the table weights the corners by those fractions.
 *
 *   1D  linear       2 corners
 *   2D  bilinear     4 corners
 *   3D  trilinear    8 corners
 *   ND  multilinear  2^N corners
 *
 * Table1D, Table2D and Table3D write their corner sums out by hand. At those
 * ranks three explicit sums cost less machinery than a fold over 2^N, and you
 * can read each weight off the line it sits on. TableND covers a fourth axis
 * and beyond, summing the same corners in the same order, so a cross-check
 * against the explicit three agrees to rounding.
 *
 * Each axis you add multiplies the storage by its breakpoint count and doubles
 * the corner count. Real schedules stop at three and spend their breakpoints
 * where the plant changes.
 *
 * @todo Tensor payload. Blocked on SqueezeView.
 *
 * SCALAR PAYLOAD ONLY. A multivariable controller's gain is a matrix, so a
 * scheduled gain needs a tensor at each grid point: a 2D schedule of 2x4
 * matrices would be one [Na, Nb, 2, 4] tensor with the corners as
 * MultiSliceViews. Extracting a corner gives [1, 1, 2, 4] where the caller
 * wants [2, 4]. SqueezeView would drop the leading ones, and it does not exist
 * yet, so this covers scalar schedules: PID gains, fuzzy membership functions.
 * A matrix payload extends this code.
 */

namespace interpolation
{

    /**
     * @brief One scheduling axis, linear interpolation.
     *
     * @tparam T     Scalar type of both the values and the scheduling variable.
     * @tparam Axis  UniformAxis or NonUniformAxis.
     */
    template <typename T, typename Axis>
    class Table1D
    {
        static constexpr my_size_t N = Axis::NumBreakpoints;

        const Axis &_axis;
        FusedTensorND<T, N> _values;

    public:
        using value_type = T;
        static constexpr my_size_t Rank = 1;

        /**
         * @param axis    Breakpoints. The table holds a reference: several
         *                tables share one axis, and the axis outlives them.
         * @param values  One value per breakpoint.
         */
        Table1D(const Axis &axis, const FusedTensorND<T, N> &values) noexcept
            : _axis(axis), _values(values) {}

        /// @brief Interpolate at @p q, using @p cursor to shorten the search.
        T at(T q, Cursor<1> &cursor) const noexcept
        {
            const auto loc = _axis.locate(q, cursor.cells[0]);

            const T w1 = loc.frac;
            const T w0 = T(1) - w1;

            return w0 * _values(loc.cell) + w1 * _values(loc.cell + 1);
        }

        /// @brief Interpolate at @p q without a cursor.
        T at(T q) const noexcept
        {
            Cursor<1> scratch;
            return at(q, scratch);
        }
    };

    /**
     * @brief Two scheduling axes, bilinear interpolation.
     *
     * The four corner weights are the products of the per-axis fractions, so a
     * query 46% along the first axis and 64% along the second weights the
     * far corner by 0.46 * 0.64.
     *
     * @tparam T      Scalar type.
     * @tparam Axis0  First axis, indexing the rows.
     * @tparam Axis1  Second axis, indexing the columns.
     */
    template <typename T, typename Axis0, typename Axis1>
    class Table2D
    {
        static constexpr my_size_t N0 = Axis0::NumBreakpoints;
        static constexpr my_size_t N1 = Axis1::NumBreakpoints;

        const Axis0 &_axis0;
        const Axis1 &_axis1;
        FusedTensorND<T, N0, N1> _values;

    public:
        using value_type = T;
        static constexpr my_size_t Rank = 2;

        Table2D(const Axis0 &axis0, const Axis1 &axis1,
                const FusedTensorND<T, N0, N1> &values) noexcept
            : _axis0(axis0), _axis1(axis1), _values(values) {}

        /// @brief Interpolate at (@p q0, @p q1), using @p cursor per axis.
        T at(T q0, T q1, Cursor<2> &cursor) const noexcept
        {
            const auto l0 = _axis0.locate(q0, cursor.cells[0]);
            const auto l1 = _axis1.locate(q1, cursor.cells[1]);

            const T a1 = l0.frac;
            const T a0 = T(1) - a1;
            const T b1 = l1.frac;
            const T b0 = T(1) - b1;

            const my_size_t i = l0.cell;
            const my_size_t j = l1.cell;

            return a0 * b0 * _values(i, j) + a1 * b0 * _values(i + 1, j) +
                   a0 * b1 * _values(i, j + 1) + a1 * b1 * _values(i + 1, j + 1);
        }

        /// @brief Interpolate without a cursor.
        T at(T q0, T q1) const noexcept
        {
            Cursor<2> scratch;
            return at(q0, q1, scratch);
        }
    };

    /**
     * @brief Three scheduling axes, trilinear interpolation.
     *
     * Eight corners. Each weight is the product of three per-axis fractions.
     *
     * @tparam T      Scalar type.
     * @tparam Axis0  First axis.
     * @tparam Axis1  Second axis.
     * @tparam Axis2  Third axis.
     */
    template <typename T, typename Axis0, typename Axis1, typename Axis2>
    class Table3D
    {
        static constexpr my_size_t N0 = Axis0::NumBreakpoints;
        static constexpr my_size_t N1 = Axis1::NumBreakpoints;
        static constexpr my_size_t N2 = Axis2::NumBreakpoints;

        const Axis0 &_axis0;
        const Axis1 &_axis1;
        const Axis2 &_axis2;
        FusedTensorND<T, N0, N1, N2> _values;

    public:
        using value_type = T;
        static constexpr my_size_t Rank = 3;

        Table3D(const Axis0 &axis0, const Axis1 &axis1, const Axis2 &axis2,
                const FusedTensorND<T, N0, N1, N2> &values) noexcept
            : _axis0(axis0), _axis1(axis1), _axis2(axis2), _values(values) {}

        /// @brief Interpolate at (@p q0, @p q1, @p q2), using @p cursor per axis.
        T at(T q0, T q1, T q2, Cursor<3> &cursor) const noexcept
        {
            const auto l0 = _axis0.locate(q0, cursor.cells[0]);
            const auto l1 = _axis1.locate(q1, cursor.cells[1]);
            const auto l2 = _axis2.locate(q2, cursor.cells[2]);

            const T a1 = l0.frac;
            const T a0 = T(1) - a1;
            const T b1 = l1.frac;
            const T b0 = T(1) - b1;
            const T c1 = l2.frac;
            const T c0 = T(1) - c1;

            const my_size_t i = l0.cell;
            const my_size_t j = l1.cell;
            const my_size_t k = l2.cell;

            return a0 * b0 * c0 * _values(i, j, k) + a1 * b0 * c0 * _values(i + 1, j, k) + a0 * b1 * c0 * _values(i, j + 1, k) +
                   a1 * b1 * c0 * _values(i + 1, j + 1, k) + a0 * b0 * c1 * _values(i, j, k + 1) + a1 * b0 * c1 * _values(i + 1, j, k + 1) +
                   a0 * b1 * c1 * _values(i, j + 1, k + 1) + a1 * b1 * c1 * _values(i + 1, j + 1, k + 1);
        }

        /// @brief Interpolate without a cursor.
        T at(T q0, T q1, T q2) const noexcept
        {
            Cursor<3> scratch;
            return at(q0, q1, q2, scratch);
        }
    };

    namespace detail
    {

        // make_index_seq recurses down to make_index_seq<0, Is...>, which holds
        // the result in a nested type. AxisPack's specialization matches
        // index_seq, so resolve the generator before passing it along.
        template <my_size_t N>
        using index_seq_t = typename make_index_seq<N>::type;

        // One reference per axis, tagged by position. The index keeps each base
        // unique when two axes share a type, so axis_slot can deduce Axis from a
        // base-class conversion.
        template <my_size_t I, typename Axis>
        struct AxisSlot
        {
            const Axis &axis;

            constexpr explicit AxisSlot(const Axis &a) noexcept : axis(a) {}
        };

        template <typename Seq, typename... Axes>
        struct AxisPack;

        // Multiple inheritance holds access to one base-class conversion at any
        // rank. A recursive list would cost an instantiation per index.
        template <my_size_t... Is, typename... Axes>
        struct AxisPack<index_seq<Is...>, Axes...> : AxisSlot<Is, Axes>...
        {
            constexpr explicit AxisPack(const Axes &...axes) noexcept
                : AxisSlot<Is, Axes>(axes)... {}
        };

        // The caller supplies I; the compiler deduces Axis from whichever base
        // matches.
        template <my_size_t I, typename Axis>
        constexpr const Axis &axis_slot(const AxisSlot<I, Axis> &slot) noexcept
        {
            return slot.axis;
        }

    } // namespace detail

    /**
     * @brief Any number of scheduling axes, multilinear interpolation.
     *
     * The rank-N generalisation of Table1D/2D/3D. A bit pattern names each of
     * the 2^N corners: bit k selects the far breakpoint on axis k, so corner 0
     * sits at the low breakpoint on every axis and corner 2^N-1 at the high one.
     * The weight multiplies frac where the bit is set and 1-frac where it is
     * clear, and the index along axis k is cell + bit.
     *
     * The fold walks the corners in ascending bit-pattern order and multiplies
     * the weights left to right, matching what Table2D and Table3D write by
     * hand. Results still differ in the last bits, because the two shapes give
     * the compiler different FMA contraction opportunities under the default
     * -ffp-contract=fast.
     *
     * Queries come in as an array so the cursor can stay last:
     *
     * @code
     * Cursor<3> cursor;
     * const double k = table.at({airspeed, altitude, mass}, cursor);
     * @endcode
     *
     * There is no variadic at(q0, q1, q2, cursor). A parameter pack cannot be
     * deduced ahead of a trailing argument, so the cursor would have to move to
     * the front, leaving two call shapes that disagree about argument order.
     *
     * @tparam T     Scalar type of the values and of every scheduling variable.
     * @tparam Axes  One axis type per scheduling variable, in grid-index order.
     */
    template <typename T, typename... Axes>
    class TableND
    {
    public:
        using value_type = T;

        static constexpr my_size_t Rank = sizeof...(Axes);

        using values_type = FusedTensorND<T, Axes::NumBreakpoints...>;

    private:
        static constexpr my_size_t NumCorners = my_size_t(1) << Rank;

        static_assert(Rank >= 1, "TableND: needs at least one axis");

        // The fold unrolls 2^Rank corner terms at compile time, and the storage
        // grows with the product of the breakpoint counts. Rank 8 already means
        // 256 terms. Past that, pick a different algorithm.
        static_assert(Rank <= 8, "TableND: rank above 8 unrolls 2^Rank corners");

        detail::AxisPack<detail::index_seq_t<Rank>, Axes...> _axes;
        values_type _values;

        // Assignment is an expression, so each axis folds without a helper.
        // Each cursor slot goes in by reference: NonUniformAxis writes back the
        // cell it settled on, ready for the next query.
        template <my_size_t... Ks>
        void locate_all(const T (&q)[Rank], Cursor<Rank> &cursor,
                        Location<T> (&loc)[Rank], index_seq<Ks...>) const noexcept
        {
            ((loc[Ks] = detail::axis_slot<Ks>(_axes).locate(q[Ks], cursor.cells[Ks])), ...);
        }

        // Bit k of Corner picks the far breakpoint on axis k: it selects frac
        // over 1-frac in the weight, and shifts the index up by one.
        template <my_size_t Corner, my_size_t... Ks>
        T corner(const Location<T> (&loc)[Rank], index_seq<Ks...>) const noexcept
        {
            const T w = (T(1) * ... * (((Corner >> Ks) & my_size_t(1)) ? loc[Ks].frac : T(1) - loc[Ks].frac));

            return w * _values((loc[Ks].cell + ((Corner >> Ks) & my_size_t(1)))...);
        }

        template <my_size_t... Cs>
        T blend(const Location<T> (&loc)[Rank], index_seq<Cs...>) const noexcept
        {
            return (T(0) + ... + corner<Cs>(loc, detail::index_seq_t<Rank>{}));
        }

    public:
        /**
         * @param axes    One axis per scheduling variable, in grid-index order.
         *                The table holds references: several tables share an
         *                axis, and the axes outlive them.
         * @param values  One value per grid point, shaped by the breakpoint counts.
         */
        TableND(const Axes &...axes, const values_type &values) noexcept
            : _axes(axes...), _values(values) {}

        /**
         * @brief Values-first overload, for class template argument deduction.
         *
         * Deduction needs the axis pack at the end of the parameter list, so
         * CTAD only works through this order. At rank 4 the alternative
         * restates the breakpoint counts the axes already carry:
         *
         * @code
         *   // without CTAD
         *   TableND<double, NonUniformAxis<double, 5>, UniformAxis<double, 4>,
         *           NonUniformAxis<double, 3>, UniformAxis<double, 3>>
         *       kq_table(airspeed, altitude, mass, mach, kq);
         *
         *   // with
         *   TableND kq_table(kq, airspeed, altitude, mass, mach);
         * @endcode
         */
        TableND(const values_type &values, const Axes &...axes) noexcept
            : _axes(axes...), _values(values) {}

        /// @brief The axis at grid position @p K.
        template <my_size_t K>
        constexpr const auto &axis() const noexcept
        {
            static_assert(K < Rank, "TableND::axis: index out of range");
            return detail::axis_slot<K>(_axes);
        }

        /// @brief Interpolate at @p q, using @p cursor to shorten each axis search.
        T at(const T (&q)[Rank], Cursor<Rank> &cursor) const noexcept
        {
            Location<T> loc[Rank];

            locate_all(q, cursor, loc, detail::index_seq_t<Rank>{});

            return blend(loc, detail::index_seq_t<NumCorners>{});
        }

        /// @brief Interpolate at @p q without a cursor.
        T at(const T (&q)[Rank]) const noexcept
        {
            Cursor<Rank> scratch;
            return at(q, scratch);
        }
    };

} // namespace interpolation
