#include "hubbardGaugeAction.hpp"

using HGA = cnxx::HubbardGaugeAction;

namespace cnxx {

    std::complex<double> HGA::eval(const Vector<std::complex<double>> &phi) {
        return blaze::sqrNorm(phi)/2./utilde;
    }

    Vector<std::complex<double>> HGA::force(const Vector<std::complex<double>> &phi) {
        return Vector<std::complex<double>>(phi.size(), 0);
    }

    std::pair<std::complex<double>, Vector<std::complex<double>>> HGA::valForce(
        const Vector<std::complex<double>> &phi) {

        return {eval(phi), force(phi)};
    }
}  // namespace cnxx

