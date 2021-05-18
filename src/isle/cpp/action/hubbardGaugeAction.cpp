#include "hubbardGaugeAction.hpp"

using HGA = isle::action::HubbardGaugeAction;
using HGAML = isle::action::HubbardGaugeActionML;

namespace isle {
    namespace action {
        std::complex<double> HGA::eval(const Vector<std::complex<double>> &phi) const {
            return (phi, phi)/2./utilde;
        }

        Vector<std::complex<double>> HGA::force(const Vector<std::complex<double>> &phi) const {
	         return -phi/utilde;
        }
        std::complex<double> HGAML::eval(const Vector<std::complex<double>> &phi) const {
            return (phi, phi)/2./utilde;
        }
        // return zero force since the NN calculates the force already (defined to suit isle format)
        Vector<std::complex<double>> HGAML::force(const Vector<std::complex<double>> &phi) const {
	             return 0*phi/utilde;
        }

    }  // namespace action
}  // namespace isle

