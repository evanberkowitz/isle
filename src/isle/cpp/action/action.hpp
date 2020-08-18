/** \file
 * \brief Abstract base action.
 */

#ifndef ACTION_ACTION_HPP
#define ACTION_ACTION_HPP

#include <memory>

#include "../math.hpp"

namespace isle {
    /// Contains all actions implemented in C++.
    namespace action {
        /// Base class for trajectory handles.
        struct BaseTrajectoryHandle
        {
            virtual ~BaseTrajectoryHandle() noexcept = default;

            // Evaluate the %Action for given auxilliary field phi.
            virtual std::complex<double> eval(const Vector<std::complex<double>> &phi) const = 0;

            /// Calculate force for given auxilliary field phi.
            virtual Vector<std::complex<double>> force(const Vector<std::complex<double>> &phi) const = 0;
        };

        /// Template for concrete implementations of trajectory handles.
        /**
         * Use this in an action to simply delegate calls to eval and force to the action objects itself.
         * Specialize the template to add custom behavior.
         *
         * \attention The base implementation depends on the action object and must not outlive it!
         */
        template <typename ActionType>
        struct TrajectoryHandle : BaseTrajectoryHandle
        {
            /// Store a reference to the action.
            explicit TrajectoryHandle(const ActionType &action)
                : _action{action}
            { }

            ~TrajectoryHandle() noexcept override = default;

            /// Evaluate the %Action on a given field phi.
            std::complex<double> eval(const Vector<std::complex<double>> &phi) const override {
                return _action.eval(phi);
            }

            /// Calculate force for a given field phi.
            Vector<std::complex<double>> force(const Vector<std::complex<double>> &phi) const override {
                return _action.force(phi);
            }

        private:
            const ActionType &_action;
        };


        /// Abstract base for Actions.
        struct Action {
            virtual ~Action() = default;

            /// Begin a new trajectory by constructing a %TrajectoryHandle object for this action.
            std::unique_ptr<BaseTrajectoryHandle> beginTrajectory() const {
                return std::unique_ptr<BaseTrajectoryHandle>{_makeTrajectoryHandle()};
            }

        protected:
            /// Construct a new trajectory handle.
            virtual BaseTrajectoryHandle *_makeTrajectoryHandle() const = 0;
        };
    }  // namespace action
}  // namespace isle

#endif  // ndef ACTION_ACTION_HPP
