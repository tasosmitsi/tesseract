#pragma once

#include "config.h"
#include "containers/array.h"
#include "helper_traits.h"
#include "simple_type_traits.h"

/**
 * @file axis.h
 * @brief Breakpoint axes for interpolated lookup tables.
 *
 * An axis owns the breakpoints along one scheduling variable and answers, for a
 * query, which cell it falls in and how far into that cell. It knows nothing
 * about what is stored at the grid points.
 *
 * Two kinds, as separate types rather than one type with a flag:
 *
 *   UniformAxis      evenly spaced, so the lookup is one divide
 *   NonUniformAxis   arbitrary breakpoints, so the lookup searches
 *
 * A table names its axis types, so the uniform case compiles to arithmetic with
 * no branch. Non-uniform is the general case — the point of scheduling is that
 * the plant behaves differently in different regions, which is what clustered
 * breakpoints express.
 *
 * Queries clamp at the domain edges. Extrapolating a gain schedule past its
 * validation range is how instability nobody predicted gets in.
 *
 * The search cursor lives outside the axis, passed in by the caller. An axis is
 * what the table was designed on; a cursor is where one caller was last time.
 * Keeping them apart makes axes immutable and shareable across threads, and
 * puts the cursor at the granularity where its locality assumption holds — two
 * control loops at different operating points would thrash a shared one.
 */

namespace interpolation
{

    /**
     * @brief Where a query landed: which cell, and how far into it.
     *
     * @p cell is the index of the lower breakpoint, so the query lies between
     * breakpoints @p cell and @p cell + 1. @p frac is 0 at the lower and 1 at
     * the upper. A clamped query gives frac 0 or 1 at the boundary cell.
     */
    template <typename T>
    struct Location
    {
        my_size_t cell;
        T frac;
    };

    /**
     * @brief Per-caller search state, one index per axis.
     *
     * Seeded at 0 and updated in place by each query. A control loop's query
     * moves slowly, so the previous cell and its neighbours are almost always
     * the answer; the full search is the fallback for a jump.
     *
     * @tparam Rank  Number of axes the table is indexed by.
     */
    template <my_size_t Rank>
    struct Cursor
    {
        Array<my_size_t, Rank> cells{};
    };

    /**
     * @brief Evenly spaced breakpoints.
     *
     * Stores an origin and a spacing rather than the breakpoints themselves, so
     * the lookup is arithmetic and there is nothing to search. The cursor is
     * accepted and ignored, so uniform and non-uniform axes are
     * interchangeable at the call site.
     *
     * @tparam T  Scalar type of the scheduling variable.
     * @tparam N  Number of breakpoints, so N-1 cells.
     */
    template <typename T, my_size_t N>
    class UniformAxis
    {
        static_assert(N >= 2, "UniformAxis: at least two breakpoints are required");

        T _origin;
        T _spacing;

    public:
        static constexpr my_size_t NumBreakpoints = N;
        static constexpr my_size_t NumCells = N - 1;
        using value_type = T;

        /**
         * @param origin   The first breakpoint.
         * @param spacing  Distance between consecutive breakpoints, > 0.
         */
        constexpr UniformAxis(T origin, T spacing) noexcept
            : _origin(origin), _spacing(spacing) {}

        constexpr T origin() const noexcept { return _origin; }
        constexpr T spacing() const noexcept { return _spacing; }

        /// @brief The i-th breakpoint.
        constexpr T breakpoint(my_size_t i) const noexcept
        {
            return _origin + T(i) * _spacing;
        }

        /**
         * @brief Locate @p q, clamping outside the domain.
         *
         * The cursor is unused — there is nothing to search — but taken so the
         * two axis types share a signature.
         */
        Location<T> locate(T q, my_size_t & /*cursor*/) const noexcept
        {
            const T scaled = (q - _origin) / _spacing;

            if (scaled <= T(0))
            {
                return {0, T(0)};
            }

            // The last cell is NumCells - 1, spanning breakpoints N-2 and N-1
            if (scaled >= T(NumCells))
            {
                return {NumCells - 1, T(1)};
            }

            const my_size_t cell = static_cast<my_size_t>(scaled);
            return {cell, scaled - T(cell)};
        }

        /// @brief Locate @p q without a cursor.
        Location<T> locate(T q) const noexcept
        {
            my_size_t ignored = 0;
            return locate(q, ignored);
        }
    };

    /**
     * @brief Arbitrary ascending breakpoints.
     *
     * Searches for the cell containing the query, starting from the cursor and
     * walking outward. For a slowly-moving query that is one or two
     * comparisons; a jump falls back to scanning from the start.
     *
     * A linear scan rather than a binary search: schedules have five to twenty
     * breakpoints, where the branch-free walk wins.
     *
     * @tparam T  Scalar type of the scheduling variable.
     * @tparam N  Number of breakpoints, so N-1 cells.
     */
    template <typename T, my_size_t N>
    class NonUniformAxis
    {
        static_assert(N >= 2, "NonUniformAxis: at least two breakpoints are required");

        Array<T, N> _breakpoints;

    public:
        static constexpr my_size_t NumBreakpoints = N;
        static constexpr my_size_t NumCells = N - 1;
        using value_type = T;

        /**
         * @param breakpoints  Ascending, strictly increasing. Not checked —
         *                     a runtime check would cost a loop per
         *                     construction to catch a mistake visible at the
         *                     call site.
         */
        constexpr explicit NonUniformAxis(const Array<T, N> &breakpoints) noexcept
            : _breakpoints(breakpoints) {}

        /// @brief Takes ownership of @p breakpoints. Same requirements as above.
        constexpr explicit NonUniformAxis(Array<T, N> &&breakpoints) noexcept
            : _breakpoints(move(breakpoints)) {}

        /// @brief Construct from the breakpoints directly, without an Array.
        template <typename... Vals>
            requires(sizeof...(Vals) == N)
        constexpr explicit NonUniformAxis(Vals... vals) noexcept
            : _breakpoints{static_cast<T>(vals)...}
        {
        }

        /// @brief The i-th breakpoint.
        constexpr T breakpoint(my_size_t i) const noexcept
        {
            return _breakpoints[i];
        }

        /**
         * @brief Locate @p q, clamping outside the domain.
         *
         * Starts at the cursor and walks toward the query, then writes the
         * cell back for the next call.
         */
        Location<T> locate(T q, my_size_t &cursor) const noexcept
        {
            if (q <= _breakpoints[0])
            {
                cursor = 0;
                return {0, T(0)};
            }

            if (q >= _breakpoints[N - 1])
            {
                cursor = NumCells - 1;
                return {NumCells - 1, T(1)};
            }

            my_size_t cell = (cursor < NumCells) ? cursor : 0;

            // Walk down while the query is below this cell
            while (cell > 0 && q < _breakpoints[cell])
            {
                --cell;
            }

            // Walk up while the query is above this cell
            while (cell + 1 < NumCells && q >= _breakpoints[cell + 1])
            {
                ++cell;
            }

            cursor = cell;

            const T lo = _breakpoints[cell];
            const T hi = _breakpoints[cell + 1];
            return {cell, (q - lo) / (hi - lo)};
        }

        /// @brief Locate @p q without a cursor, searching from the start.
        Location<T> locate(T q) const noexcept
        {
            my_size_t scratch = 0;
            return locate(q, scratch);
        }
    };

} // namespace interpolation
