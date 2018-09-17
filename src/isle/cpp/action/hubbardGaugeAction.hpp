/** \file
 * \brief Gauge part of Hubbard action.
 */

#ifndef ACTION_HUBBARD_GAUGE_ACTION_HPP
#define ACTION_HUBBARD_GAUGE_ACTION_HPP

#include "action.hpp"

namespace isle {
    namespace action {
        /// Pure gauge action for Hubbard model.
        /**
         * The action is
         \f[
         S_{\mathrm{HGA}} = \frac{1}{2\tilde{U}} \textstyle\sum_{x,t}\,\phi^2_{xt}.
         \f]
        */
        class HubbardGaugeAction : public Action {
        public:
            double utilde;  ///< Parameter \f$\tilde{U}\f$.

            /// Set \f$\tilde{U}\f$.
            explicit HubbardGaugeAction(const double utilde_) : utilde{utilde_} { }

            HubbardGaugeAction(const HubbardGaugeAction &other) = default;
            HubbardGaugeAction &operator=(const HubbardGaugeAction &other) = default;
            HubbardGaugeAction(HubbardGaugeAction &&other) = default;
            HubbardGaugeAction &operator=(HubbardGaugeAction &&other) = default;
            ~HubbardGaugeAction() override = default;

            /// Evaluate the %Action for given auxilliary field phi.
            std::complex<double> eval(const Vector<std::complex<double>> &phi) const override;

            /// Calculate force for given auxilliary field phi.
            Vector<std::complex<double>> force(const Vector<std::complex<double>> &phi) const override;
        };
    }  // namespace action
}  // namespace isle

#endif  // ndef ACTION_HUBBARD_GAUGE_ACTION_HPP
