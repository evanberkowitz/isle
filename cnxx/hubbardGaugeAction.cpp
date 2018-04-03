#include "hubbardGaugeAction.hpp"

using HGA = cnxx::HubbardGaugeAction;

namespace cnxx {

    std::complex<double> HGA::eval(const Vector<std::complex<double>> &phi) {
        return blaze::sqrNorm(phi)/2./utilde;
    }

    Vector<std::complex<double>> HGA::force(const Vector<std::complex<double>> &phi) {
        return -phi/utilde;
    }
}  // namespace cnxx

