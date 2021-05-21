#include "hubbardGaugeAction.hpp"

using HGA = isle::action::HubbardGaugeAction;

namespace isle {
    namespace action {
        std::complex<double> HGA::eval(const Vector<std::complex<double>> &phi) const {
            return (phi, phi)/2./utilde;
        }

        Vector<std::complex<double>> HGA::force(const Vector<std::complex<double>> &phi) const {
	         return -phi/utilde;
        }

    }  // namespace action
}  // namespace isle

