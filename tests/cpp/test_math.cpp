#include <cmath>
#include <random>

#include "catch2/catch.hpp"
#include <math.hpp>

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
