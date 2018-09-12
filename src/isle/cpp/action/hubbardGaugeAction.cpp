#include "hubbardGaugeAction.hpp"

using HGA = cnxx::action::HubbardGaugeAction;

namespace cnxx {
    namespace action {
        std::complex<double> HGA::eval(const Vector<std::complex<double>> &phi) const {
            return (blaze::conj(phi), phi)/2./utilde;
        }

        Vector<std::complex<double>> HGA::force(const Vector<std::complex<double>> &phi) const {
            return -phi/utilde;
        }
    }  // namespace action
}  // namespace cnxx

