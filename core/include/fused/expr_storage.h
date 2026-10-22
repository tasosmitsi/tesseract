#pragma once

#include "simple_type_traits.h"
#include "expression_traits/expression_traits.h"

/**
 * @brief How an expression node or view stores one of its operands.
 *
 * Tensors own their elements, so nodes refer to them: copying one would
 * duplicate its whole buffer for every expression it appears in. Everything
 * else, expression nodes and views, is a few references plus compile-time
 * shape, so nodes keep their own copy. The copy lives as long as the node
 * holding it, which is what lets an expression outlive the statement that
 * built it:
 *
 *   auto e = (a + b) * (a - b);   // both sums are copied into e
 *   out = e;                      // fine: e no longer refers to dead temporaries
 *
 * The remaining limit is a temporary tensor used as an operand. It is still
 * held by reference, so it dies at the end of the statement:
 *
 *   auto e = a + FusedMatrix<T, N, N>::matmul(b, c);   // dangles
 *
 * expression::traits<E>::OwnsMemory picks the storage. Every tensor, view and
 * expression node sets it in its traits specialization. The primary traits
 * template has no body, so a new type that forgot its specialization stops
 * the build here ("incomplete type expression::traits<...>") instead of
 * getting a default that is wrong for it: holding a temporary view by
 * reference would dangle, and copying a tensor would duplicate its buffer.
 *
 * @tparam E  The operand type.
 */
template <typename E>
using ExprStorage = conditional_t<expression::traits<E>::OwnsMemory, const E &, const E>;
