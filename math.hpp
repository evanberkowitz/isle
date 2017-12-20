#ifndef MATH_HPP
#define MATH_HPP

#include <blaze/Math.h>

template <typename ET>
using Vector = blaze::DynamicVector<ET>;

template <typename ET>
using Vec3 = blaze::StaticVector<ET, 3>;

template <typename ET>
using Matrix = blaze::DynamicMatrix<ET>;

template <typename ET>
using SparseMatrix = blaze::CompressedMatrix<ET>;

template <typename ET>
using IdMatrix = blaze::IdentityMatrix<ET>;

#endif  // ndef MATH_HPP
