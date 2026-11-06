#pragma once

#include <catch_amalgamated.hpp>

#include "config.h"

// Forward and back for every element of a layout, then every offset of its
// buffer: the layout must accept its own elements and nothing else.
template <typename L>
void check_two_way(my_size_t buffer_size)
{
    my_size_t coords[L::NumDims];
    my_size_t back[L::NumDims];
    my_size_t mismatches = 0;

    for (my_size_t flat = 0; flat < L::LogicalSize; ++flat)
    {
        L::logical_flat_to_logical_coords(flat, coords);

        const my_size_t offset = L::logical_coords_to_physical_flat(coords);
        mismatches += L::logical_flat_to_physical_flat(flat) != offset;
        mismatches += L::logical_coords_to_logical_flat(coords) != flat;
        mismatches += !L::is_physical_flat_in_layout(offset);

        L::physical_flat_to_logical_coords(offset, back);
        for (my_size_t i = 0; i < L::NumDims; ++i)
        {
            mismatches += back[i] != coords[i];
        }
        mismatches += L::physical_flat_to_logical_flat(offset) != flat;
    }

    my_size_t accepted = 0;
    for (my_size_t offset = 0; offset < buffer_size; ++offset)
    {
        accepted += L::is_physical_flat_in_layout(offset);
    }

    CHECK(mismatches == 0);
    CHECK(accepted == L::LogicalSize);
}
