#pragma once

#include "config.h"
#include "containers/array.h"
#include "helper_traits.h"

template <bool Enabled, my_size_t... Vals>
struct PermValidation
{
    static constexpr bool unique = true;
    static constexpr my_size_t max_val = 0;
    static constexpr my_size_t min_val = 0;
};

template <my_size_t... Vals>
struct PermValidation<true, Vals...>
{
    static constexpr bool unique = all_unique<Vals...>();
    static constexpr my_size_t max_val = max_value<Vals...>();
    static constexpr my_size_t min_val = min_value<Vals...>();
};

/**
 * @brief Compile-time strided layout with optional permutation.
 *
 * @tparam Policy  Padding policy (e.g., SimdPaddingPolicy<T, Dims...>)
 * @tparam Perm    Optional permutation indices (empty = identity)
 *
 * All computations happen at compile-time. Zero runtime overhead.
 *
 * Usage:
 *   StridedLayoutConstExpr<Policy>           // no permutation (identity)
 *   StridedLayoutConstExpr<Policy, 1, 0>     // transposed
 *   StridedLayoutConstExpr<Policy, 2, 0, 1>  // custom permutation
 */
template <typename Policy, my_size_t... Perm>
struct StridedLayoutConstExpr
{
    static constexpr my_size_t NumDims = Policy::NumDims;
    static constexpr my_size_t LogicalSize = Policy::LogicalSize;
    static constexpr my_size_t PhysicalSize = Policy::PhysicalSize;
    static constexpr bool IsPermProvided = sizeof...(Perm) > 0;

private:
    using PermCheck = PermValidation<IsPermProvided, Perm...>;

public:
    static_assert(!IsPermProvided || sizeof...(Perm) == NumDims,
                  "Permutation must match number of dimensions");

    static_assert(!IsPermProvided || PermCheck::unique,
                  "Permutations must be unique");

    static_assert(!IsPermProvided || PermCheck::max_val < NumDims,
                  "Max value of permutation pack must be less than number of dimensions");

    static_assert(!IsPermProvided || PermCheck::min_val == 0,
                  "Min value of permutation pack is not equal to 0");

    /** Identity or provided permutation */
    static constexpr Array<my_size_t, NumDims> computePermArray() noexcept
    {
        if constexpr (IsPermProvided)
        {
            return Array<my_size_t, NumDims>{Perm...};
        }
        else
        {
            Array<my_size_t, NumDims> result{};
            for (my_size_t i = 0; i < NumDims; ++i)
            {
                result[i] = i;
            }
            return result;
        }
    }

    static constexpr Array<my_size_t, NumDims> PermArray = computePermArray();

    /** Inverse permutation array for mapping back to original layout.
     * Only computed if permutation is provided, otherwise it's just the identity.
     */
    static constexpr Array<my_size_t, NumDims> computeInversePermArray() noexcept
    {
        Array<my_size_t, NumDims> result{};
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            result[PermArray[i]] = i;
        }
        return result;
    }

    static constexpr Array<my_size_t, NumDims> InversePermArray = computeInversePermArray();

    // ========================================================================
    // LOGICAL DIMENSIONS (permuted)
    // ========================================================================

    static constexpr Array<my_size_t, NumDims> computeLogicalDims() noexcept
    {
        Array<my_size_t, NumDims> result{};
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            result[i] = Policy::LogicalDims[PermArray[i]];
        }
        return result;
    }

    static constexpr Array<my_size_t, NumDims> LogicalDims = computeLogicalDims();

    // ========================================================================
    // BASE STRIDES (unpermuted, for physical decomposition)
    // ========================================================================

    static constexpr Array<my_size_t, NumDims> computeBaseStrides() noexcept
    {
        Array<my_size_t, NumDims> result{};
        result[NumDims - 1] = 1;
        for (my_size_t i = NumDims - 1; i > 0; --i)
        {
            result[i - 1] = result[i] * Policy::PhysicalDims[i];
        }
        return result;
    }

    static constexpr Array<my_size_t, NumDims> BaseStrides = computeBaseStrides();

    // ========================================================================
    // PHYSICAL STRIDES (permuted)
    // ========================================================================

    /**
     * Compute physical strides with permutation applied.
     *
     * Two steps:
     *   1. Compute base strides from unpermuted physical dims (above) BaseStrides
     *   2. Permute the strides (not the dims!)
     *
     * Why not compute strides from permuted physical dims?
     * Memory layout never changes — a permuted view just accesses it differently.
     * We must permute the stride VALUES, not recompute from permuted dims.
     */
    static constexpr Array<my_size_t, NumDims> computeStrides() noexcept
    {
        // Permute the BaseStrides
        Array<my_size_t, NumDims> result{};
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            result[i] = BaseStrides[PermArray[i]];
        }
        return result;
    }

    static constexpr Array<my_size_t, NumDims> Strides = computeStrides();

    // ========================================================================
    // LOGICAL STRIDES (for flat index decomposition)
    // ========================================================================

    static constexpr Array<my_size_t, NumDims> computeLogicalStrides() noexcept
    {
        Array<my_size_t, NumDims> result{};
        result[NumDims - 1] = 1;
        for (my_size_t i = NumDims - 1; i > 0; --i)
        {
            result[i - 1] = result[i] * LogicalDims[i];
        }
        return result;
    }

    static constexpr Array<my_size_t, NumDims> LogicalStrides = computeLogicalStrides();

    // ========================================================================
    // DIMENSION QUERIES
    // ========================================================================

    FORCE_INLINE static constexpr my_size_t num_dims() noexcept { return NumDims; }

    FORCE_INLINE static constexpr my_size_t logical_dim(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return LogicalDims.at(i);
    }

    FORCE_INLINE static constexpr my_size_t stride(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Strides.at(i);
    }

    // ========================================================================
    // INDEX CONVERSION
    // ========================================================================

    /**
     * Convert logical flat index to physical offset.
     *
     * Uses forward iteration + subtraction to minimize divisions.
     * Only N divisions instead of 2N with reverse + modulo.
     */
    FORCE_INLINE static constexpr my_size_t logical_flat_to_physical_flat(my_size_t logical_flat) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        if (logical_flat >= LogicalSize)
        {
            MyErrorHandler::error("logical_flat_to_physical_flat: logical_flat index out of bounds");
        }
        my_size_t offset = 0;
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            const my_size_t s = LogicalStrides[i];
            const my_size_t idx = logical_flat / s;
            logical_flat -= idx * s;
            offset += idx * Strides[i];
        }
        return offset;
    }

    FORCE_INLINE static constexpr bool is_logical_index_in_bounds(const my_size_t (&indices)[NumDims]) noexcept
    {
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            if (indices[i] >= LogicalDims[i])
                return false;
        }
        return true;
    }

    /** Array multi-index to physical offset (bounds-checked) using */
    FORCE_INLINE static constexpr my_size_t logical_coords_to_physical_flat(const my_size_t (&indices)[NumDims]) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        if (!is_logical_index_in_bounds(indices))
        {
            MyErrorHandler::error("logical_coords_to_physical_flat: index out of bounds for logical dimension");
        }

        my_size_t flat = 0;
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            flat += indices[i] * Strides[i];
        }
        return flat;
    }

    /** Variadic multi-index to physical offset (bounds-checked against logical dims) */
    template <typename... Indices>
    FORCE_INLINE static constexpr my_size_t logical_coords_to_physical_flat(Indices... indices) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        static_assert(sizeof...(Indices) == NumDims, "Wrong number of indices");
        my_size_t idx_arr[] = {static_cast<my_size_t>(indices)...};
        return logical_coords_to_physical_flat(idx_arr);
    }

    /**
     * Logical flat index to logical coordinates.
     *
     * Decomposes a flat index into multi-dimensional coordinates
     * using LogicalStrides (row-major order in logical space).
     *
     * Input must be in range [0, LogicalSize).
     *
     * Example: 3x4 matrix, logical flat 7:
     *   LogicalStrides = [4, 1]
     *   7 / 4 = 1, 7 - 4 = 3
     *   3 / 1 = 3
     *   → coords (1, 3)
     */
    FORCE_INLINE static constexpr void logical_flat_to_logical_coords(my_size_t logical_flat, my_size_t (&indices)[NumDims]) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        // check logical_flat is in range
        if (logical_flat >= LogicalSize)
        {
            MyErrorHandler::error("logical_flat_to_logical_coords: logical_flat index out of bounds");
        }
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            const my_size_t s = LogicalStrides[i];
            const my_size_t idx = logical_flat / s;
            indices[i] = idx;
            logical_flat -= idx * s;
        }
    }

    /**
     * Physical flat index to physical coordinates.
     *
     * Decomposes a flat index into multi-dimensional coordinates
     * using BaseStrides (row-major order in physical memory).
     *
     * Input must be in range [0, PhysicalSize).
     *
     * WARNING: If physical_flat is a padding location, the returned
     * coords will also be in the padding area.
     *
     * Example: 2x3 matrix padded to 2x4, physical flat 7:
     *   BaseStrides = [4, 1]
     *   7 / 4 = 1, 7 - 4 = 3
     *   3 / 1 = 3
     *   → coords (1, 3)  ← column 3 is padding (LogicalDims[1] = 3)
     */
    FORCE_INLINE static constexpr void physical_flat_to_physical_coords(my_size_t physical_flat, my_size_t (&indices)[NumDims]) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        // check physical_flat is in range
        if (physical_flat >= PhysicalSize)
        {
            MyErrorHandler::error("physical_flat_to_physical_coords: physical_flat index out of bounds");
        }

        for (my_size_t i = 0; i < NumDims; ++i)
        {
            const my_size_t s = BaseStrides[i];
            const my_size_t idx = physical_flat / s;
            indices[i] = idx;
            physical_flat -= idx * s;
        }
    }

    // ========================================================================
    // REVERSE CONVERSION
    // ========================================================================

    /**
     * Physical flat index to logical coordinates.
     *
     * Decomposes a flat index into physical coordinates using BaseStrides,
     * then applies permutation to get logical coordinates.
     *
     * Input must be in range [0, PhysicalSize).
     *
     * WARNING: If physical_flat is a padding location, the returned
     * logical coords will be out-of-bounds (coord >= LogicalDims[i]).
     *
     * Example: 2x3 matrix padded to 2x4, transposed [1,0], physical flat 5:
     *   BaseStrides = [4, 1], PermArray = [1, 0]
     *   Step 1 - decompose with BaseStrides:
     *     5 / 4 = 1, 5 - 4 = 1
     *     1 / 1 = 1
     *     → physical coords (1, 1)
     *   Step 2 - apply permutation:
     *     logical[0] = physical[PermArray[0]] = physical[1] = 1
     *     logical[1] = physical[PermArray[1]] = physical[0] = 1
     *     → logical coords (1, 1)
     */
    FORCE_INLINE static constexpr void physical_flat_to_logical_coords(my_size_t physical_flat, my_size_t (&indices)[NumDims]) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        if (physical_flat >= PhysicalSize)
        {
            MyErrorHandler::error("physical_flat_to_logical_coords: physical_flat index out of bounds");
        }
        // Step 1: Decompose into physical (unpermuted) coords
        my_size_t physical_coords[NumDims];
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            const my_size_t s = BaseStrides[i];
            const my_size_t idx = physical_flat / s;
            physical_coords[i] = idx;
            physical_flat -= idx * s;
        }

        // Step 2: Apply inverse permutation
        // logical[i] = physical[PermArray[i]]
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            indices[i] = physical_coords[PermArray[i]];
        }
    }
};
