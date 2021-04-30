#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"

#include <pybind11/embed.h>

int main(int argc, char *argv[]) {
  // Launch the Python interpreter to ensure that logging is possible.
  pybind11::scoped_interpreter guard{};
  return Catch::Session().run(argc, argv);
}