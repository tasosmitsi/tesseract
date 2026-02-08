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

    static_assert(!IsPermProvided || sizeof...(Perm) == NumDims,
                  "Permutation must match number of dimensions");

    static_assert(!IsPermProvided || PermCheck::unique,
                  "Permutations must be unique");

    static_assert(!IsPermProvided || PermCheck::max_val < NumDims,
                  "Max value of permutation pack must be less than number of dimensions");

    static_assert(!IsPermProvided || PermCheck::min_val == 0,
                  "Min value of permutation pack is not equal to 0");

    /**
     * @brief Compute the permutation array at compile-time.
     * If permutation is provided, use it. Otherwise, generate identity.
     *
     * @return constexpr Array<my_size_t, NumDims>
     */
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

    /**
     * @brief Permutation array.
     *
     */
    static constexpr Array<my_size_t, NumDims> PermArray = computePermArray();

    /**
     * @brief Compute the inverse permutation array at compile-time.
     * Inverse permutation array for mapping back to original layout.
     * Only computed if permutation is provided, otherwise it's just the identity.
     *
     * @return constexpr Array<my_size_t, NumDims>
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

    /**
     * @brief Inverse permutation array.
     *
     */
    static constexpr Array<my_size_t, NumDims> InversePermArray = computeInversePermArray();

    /**
     * @brief Compute logical dimensions with permutation applied.
     *
     * @return constexpr Array<my_size_t, NumDims>
     */
    static constexpr Array<my_size_t, NumDims> computeLogicalDims() noexcept
    {
        Array<my_size_t, NumDims> result{};
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            result[i] = Policy::LogicalDims[PermArray[i]];
        }
        return result;
    }

    /**
     * @brief Logical dimensions with permutation applied.
     *
     */
    static constexpr Array<my_size_t, NumDims> LogicalDims = computeLogicalDims();

    /**
     * @brief Compute base strides for physical decomposition (unpermuted).
     *
     * @return constexpr Array<my_size_t, NumDims>
     */
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

    /**
     * @brief Base strides for physical decomposition (unpermuted).
     *
     */
    static constexpr Array<my_size_t, NumDims> BaseStrides = computeBaseStrides();

    /**
     * @brief Compute physical strides with permutation applied from base strides.
     *
     * @return constexpr Array<my_size_t, NumDims>
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

    /**
     * @brief Physical strides with permutation applied.
     *
     */
    static constexpr Array<my_size_t, NumDims> Strides = computeStrides();

    /**
     * @brief Compute logical strides for flat index decomposition.
     *
     * @return constexpr Array<my_size_t, NumDims>
     */
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

    /**
     * @brief Logical strides for flat index decomposition.
     *
     */
    static constexpr Array<my_size_t, NumDims> LogicalStrides = computeLogicalStrides();

public:
    /**
     * @brief Get number of dimensions.
     *
     * @return constexpr my_size_t
     */
    FORCE_INLINE static constexpr my_size_t num_dims() noexcept { return NumDims; }

    /**
     * @brief Get permutation at dimension i.
     *
     * @param i
     * @return constexpr my_size_t
     * @throws if i >= NumDims
     */
    FORCE_INLINE static constexpr my_size_t perm_array(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return PermArray.at(i);
    }

    /**
     * @brief Get inverse permutation at dimension i.
     *
     * @param i
     * @return constexpr my_size_t
     * @throws if i >= NumDims
     */
    FORCE_INLINE static constexpr my_size_t inverse_perm_array(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return InversePermArray.at(i);
    }

    /**
     * @brief Get logical dimension at index i (with permutation applied).
     *
     * @param i
     * @return constexpr my_size_t
     * @throws if i >= NumDims
     */
    FORCE_INLINE static constexpr my_size_t logical_dim(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return LogicalDims.at(i);
    }

    /**
     * @brief Get base stride at dimension i (unpermuted, for physical decomposition).
     *
     * @param i
     * @return constexpr my_size_t
     * @throws if i >= NumDims
     */
    FORCE_INLINE static constexpr my_size_t base_stride(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return BaseStrides.at(i);
    }

    /**
     * @brief Get physical stride at dimension i (with permutation applied).
     *
     * @param i
     * @return constexpr my_size_t
     * @throws if i >= NumDims
     */
    FORCE_INLINE static constexpr my_size_t stride(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Strides.at(i);
    }

    /**
     * @brief Get logical stride at dimension i (for flat index decomposition).
     *
     * @param i
     * @return constexpr my_size_t
     * @throws if i >= NumDims
     */
    FORCE_INLINE static constexpr my_size_t logical_stride(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return LogicalStrides.at(i);
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

    /**
     * @brief Check if logical multi-index is in bounds.
     *
     * @return constexpr bool
     */
    FORCE_INLINE static constexpr bool is_logical_index_in_bounds(const my_size_t (&indices)[NumDims]) noexcept
    {
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            if (indices[i] >= LogicalDims[i])
                return false;
        }
        return true;
    }

    /**
     * @brief Logical coordinates (Array multi-index) to physical flat index (bounds-checked).
     *
     * @return constexpr my_size_t
     * @throws if any index is out of bounds for its logical dimension
     */
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

    /**
     * @brief Logical coordinates (variadic multi-index) to physical flat index (bounds-checked).
     *
     * @tparam Indices
     * @param indices
     * @return constexpr my_size_t
     * @throws if any index is out of bounds for its logical dimension
     */
    template <typename... Indices>
    FORCE_INLINE static constexpr my_size_t logical_coords_to_physical_flat(Indices... indices) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        static_assert(sizeof...(Indices) == NumDims, "Wrong number of indices");
        my_size_t idx_arr[] = {static_cast<my_size_t>(indices)...};
        return logical_coords_to_physical_flat(idx_arr);
    }

    /**
     * @brief Logical flat index to logical coordinates.
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
     *
     * @param logical_flat Input flat index in logical space
     * @param indices Output parameter for multi-dimensional coordinates
     * @return constexpr void
     * @throws if logical_flat is out of bounds for LogicalSize
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
     * @brief Physical flat index to physical coordinates.
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
     *
     * @param physical_flat
     * @param indices Output parameter for multi-dimensional coordinates
     * @return constexpr void
     * @throws if physical_flat is out of bounds for PhysicalSize
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
     * @brief Physical flat index to logical coordinates.
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
     * @param physical_flat
     * @param indices Output parameter for logical multi-dimensional coordinates
     * @return constexpr void
     * @throws if physical_flat is out of bounds for PhysicalSize
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
