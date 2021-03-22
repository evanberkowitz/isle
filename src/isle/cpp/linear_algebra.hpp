#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <complex>

#include <blaze/math/dense/CustomMatrix.h>
#include <blaze/math/dense/CustomVector.h>
#include <blaze/math/dense/DenseMatrix.h>
#include <blaze/math/dense/DenseVector.h>
#include <blaze/math/expressions/MatEvalExpr.h>
#include <blaze/math/sparse/CompressedMatrix.h>
#include <blaze/math/sparse/IdentityMatrix.h>

#include "tmp.hpp"

#ifdef USE_CUDA

#include "cuda_helper.cuh"
#include <cuda_runtime_api.h>

namespace isle {
template <typename T> auto *allocate_managed(const size_t size) {
  T *ptr = nullptr;
  CHECK_CU_ERR(
      cudaMallocManaged(reinterpret_cast<void **>(&ptr), size * sizeof(T)));
  return ptr;
}

template <typename T> void free_managed(T *ptr) { CHECK_CU_ERR(cudaFree(ptr)); }

template <typename ET>
class Vector : public blaze::CustomVector<ET, blaze::aligned, blaze::unpadded> {
public:
  explicit Vector()
      : blaze::CustomVector<ET, blaze::aligned, blaze::unpadded>{},
        _buffer{nullptr} {}

  explicit Vector(const size_t size)
      : blaze::CustomVector<ET, blaze::aligned, blaze::unpadded>{},
        _buffer{allocate_managed<ET>(size)} {
    try {
      this->reset(_buffer, size);
    } catch (std::invalid_argument &) {
      free_managed(_buffer);
      throw;
    }
  }

  explicit Vector(const size_t size, const ET &val) : Vector{size} {
    std::fill(_buffer, _buffer + size, val);
  }

  template <typename Other>
  explicit Vector(const size_t size, const Other *const vals) : Vector{size} {
    std::copy(vals, vals + size, _buffer);
  }

  Vector(const Vector &other) : Vector{other.size()} {
    CHECK_CU_ERR(
        cudaMemcpy(_buffer, other._buffer, other.size(), cudaMemcpyDefault));
  }

  Vector &operator=(const Vector &other) {
    if (other.size() != this->size()) {
      free_managed(_buffer);
      _buffer = allocate_managed<ET>(other.size());
      try {
        this->reset(_buffer, other.size());
      } catch (std::invalid_argument &) {
        free_managed(_buffer);
        throw;
      }
    }
    CHECK_CU_ERR(
        cudaMemcpy(_buffer, other._buffer, other.size(), cudaMemcpyDefault));
    return *this;
  }

  template <typename E> Vector(const E &expression) : Vector{} {
    *this = blaze::eval(expression);
  }

  Vector(Vector &&other) noexcept = default;
  Vector &operator=(Vector &&other) noexcept = default;

  ~Vector() noexcept { free_managed(_buffer); }

  template <typename NewType> struct Rebind { using Other = Vector<NewType>; };

private:
  ET *_buffer;
};

template <typename ET>
class Matrix : public blaze::CustomMatrix<ET, blaze::aligned, blaze::unpadded> {
public:
  explicit Matrix()
      : blaze::CustomMatrix<ET, blaze::aligned, blaze::unpadded>{},
        _buffer{nullptr} {}

  explicit Matrix(const size_t nrow, const size_t ncol)
      : blaze::CustomMatrix<ET, blaze::aligned, blaze::unpadded>{},
        _buffer{allocate_managed<ET>(nrow * ncol)} {
    try {
      this->reset(_buffer, nrow, ncol);
    } catch (std::invalid_argument &) {
      free_managed(_buffer);
      throw;
    }
  }

  explicit Matrix(const size_t nrow, const size_t ncol, const ET &val)
      : Matrix{nrow, ncol} {
    std::fill(_buffer, _buffer + nrow * ncol, val);
  }

  template <typename Other>
  explicit Matrix(const size_t nrow, const size_t ncol, const Other *const vals)
      : Matrix{nrow, ncol} {
    std::copy(vals, vals + nrow * ncol, _buffer);
  }

  Matrix(const Matrix &other) : Matrix{other.rows(), other.columns()} {
    CHECK_CU_ERR(cudaMemcpy(_buffer, other._buffer,
                            other.rows() * other.columns(), cudaMemcpyDefault));
  }

  Matrix &operator=(const Matrix &other) {
    if (other.rows() * other.columns() != this->rows() * this->columns()) {
      free_managed(_buffer);
      _buffer = allocate_managed<ET>(other.rows() * other.columns());
      try {
        this->reset(_buffer, other.rows(), other.columns());
      } catch (std::invalid_argument &) {
        free_managed(_buffer);
        throw;
      }
    }
    CHECK_CU_ERR(cudaMemcpy(_buffer, other._buffer,
                            other.rows() * other.columns(), cudaMemcpyDefault));
    return *this;
  }

  Matrix(const blaze::CompressedMatrix<ET> &other)
      : Matrix{blaze::DynamicMatrix<ET>{other}} {}

  template <typename E> Matrix(const E &expression) : Matrix{} {
    *this = blaze::eval(expression);
  }

  Matrix(Matrix &&other) noexcept = default;
  Matrix &operator=(Matrix &&other) noexcept = default;

  ~Matrix() noexcept { free_managed(_buffer); }

  void resize(const size_t nrow, const size_t ncol,
              const bool preserve = true) {
    Matrix aux{nrow, ncol};
    if (preserve) {
      CHECK_CU_ERR(cudaMemcpy(aux._buffer, this->_buffer, nrow * ncol,
                              cudaMemcpyDefault));
    }
    *this = std::move(aux);
  }

  template <typename NewType> struct Rebind { using Other = Matrix<NewType>; };

private:
  ET *_buffer;
};

} // namespace isle

#else

namespace isle {
/**
 * \brief A generic dense vector.
 * \tparam ET Element Type
 */
template <typename ET> using Vector = blaze::DynamicVector<ET>;

/**
 * \brief A generic dense matrix.
 * \tparam ET Element Type
 */
template <typename ET> using Matrix = blaze::DynamicMatrix<ET>;

} // namespace isle

#endif // def USE_CUDA

namespace isle {
/**
 * \brief Holds spatial coordinates.
 * \tparam ET Element Type
 */
template <typename ET> using Vec3 = blaze::StaticVector<ET, 3>;

/**
 * \brief A generic sparse matrix.
 * \tparam ET Element Type
 */
template <typename ET> using SparseMatrix = blaze::CompressedMatrix<ET>;

/**
 * \brief An identity matrix.
 * \tparam ET Element Type
 */
template <typename ET> using IdMatrix = blaze::IdentityMatrix<ET>;

// Convenience aliases.
using DVector = Vector<double>;
using CDVector = Vector<std::complex<double>>;

using DMatrix = Matrix<double>;
using CDMatrix = Matrix<std::complex<double>>;

using DSparseMatrix = SparseMatrix<double>;
using CDSparseMatrix = SparseMatrix<std::complex<double>>;

/// Get the element type of a linear algebra type (vector, matrix).
/**
 * Falls back to given type if it is arithmetic or std::complex.
 * Causes failure of a static assertion if the type is not recognized.
 *
 * \see ValueType %ElementType does not retrieve the value type from a compund
 * type like std::complex but only operates on collections of elemental
 * variables.
 */
template <typename T, typename = void> struct ElementType {
  static_assert(tmp::AlwaysFalse_v<T>, "Cannot deduce element type.");
};

/// Overload for arithmetic types and std::complex.
template <typename T>
struct ElementType<
    T, std::enable_if_t<std::is_arithmetic<T>::value ||
                        tmp::IsSpecialization<std::complex, T>::value>> {
  using type = T; ///< Deduced element type.
};

/// Overload for blaze::DynamicVector.
template <typename ET, bool TF>
struct ElementType<blaze::DynamicVector<ET, TF>> {
  using type = ET; ///< Deduced element type.
};

/// Overload for blaze::CustomVector.
template <typename ET> struct ElementType<Vector<ET>> {
  using type = ET; ///< Deduced element type.
};

/// Overload for blaze::DynamicMatrix.
template <typename ET, bool SO>
struct ElementType<blaze::DynamicMatrix<ET, SO>> {
  using type = ET; ///< Deduced element type.
};

/// Overload for blaze::CustomMatrix.
template <typename ET> struct ElementType<Matrix<ET>> {
  using type = ET; ///< Deduced element type.
};

/// Overload for blaze::SparseMatrix.
template <typename ET, bool SO>
struct ElementType<blaze::CompressedMatrix<ET, SO>> {
  using type = ET; ///< Deduced element type.
};

/// Convenience alias for ElementType.
template <typename T> using ElementType_t = typename ElementType<T>::type;
} // namespace isle

#endif // LINEAR_ALGEBRA_HPP
