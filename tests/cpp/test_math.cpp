#include <cmath>
#include <random>

#include "catch2/catch.hpp"

#include <math.hpp>

#ifdef USE_CUDA
#include <math.cuh>
#endif

namespace {
auto makeRandomCDMatrix(const size_t n, std::mt19937 &rng) {
  isle::CDMatrix mat(n, n);
  std::uniform_real_distribution<double> dist{-5.0, 5.0};
  std::generate(mat.data(), std::next(mat.data(), static_cast<ptrdiff_t>(n * n)),
                [&]() mutable { return std::complex{dist(rng), dist(rng)}; });
  return mat;
}
} // namespace

TEST_CASE("expmsym", "[diagonal]") {
  const auto n = GENERATE(2ul, 3ul, 5ul, 10ul);

  std::mt19937 rng{n};
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  isle::DMatrix mat(n, n, 0.0);
  isle::DMatrix expected(n, n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    mat(i, i) = dist(rng);
    expected(i, i) = std::exp(mat(i, i));
  }

  const auto expm = isle::expmSym(mat);
  for (size_t i = 0; i < n * n; ++i) {
    REQUIRE(expm.data()[i] == Approx(expected.data()[i]));
  }
}

#ifdef USE_CUDA
TEST_CASE("matmul", "[GPU]") {
  const auto n = GENERATE(2ul, 3ul, 5ul, 10ul, 34ul);
  std::mt19937 rng{3*n};

  auto a = makeRandomCDMatrix(n, rng);
  auto b = makeRandomCDMatrix(n, rng);
  auto expected = isle::evaluate(a * b);
  auto actual = isle::mult_CDMatrix_wrapper(a, b, a.rows());

  for (size_t i = 0; i < n * n; ++i) {
    REQUIRE(actual.data()[i].real() == Approx(expected.data()[i].real()));
    REQUIRE(actual.data()[i].imag() == Approx(expected.data()[i].imag()));
  }
}
#endif