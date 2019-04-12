/** \file
 * \brief Sum of other actions.
 */

#ifndef ACTION_SUM_ACTION_HPP
#define ACTION_SUM_ACTION_HPP

#include <vector>

#include "../core.hpp"
#include "action.hpp"

namespace isle {
    namespace action {
        /// Sum of several actions.
        /**
         * Stores references to arbitrary instances of derived types of Action.
         * Calling SumAction::eval() evaluates all actions and adds up the results;
         * similarily for SumAction::force().
         *
         * \attention This is a view type. It stores references to the actions
         *            passed to it but does not own them. The user is responsible
         *            for freeing any resources appropriately.
         */
        class SumAction : public Action{
        public:
            SumAction() = default;
            SumAction(const SumAction &other) = default;
            SumAction &operator=(const SumAction &other) = default;
            SumAction(SumAction &&other) noexcept = default;
            SumAction &operator=(SumAction &&other) noexcept = default;
            ~SumAction() noexcept = default;

            /// Add an action to the collection.
            /**
             * \param action Pointer to the action to reference.
             * \returns `action`.
             */
            Action *add(Action *action);

            /// Access an action via its index.
            Action *operator[](std::size_t idx);
            /// Access an action via its index.
            const Action *operator[](std::size_t idx) const;

            /// Return the number of actions currently referenced.
            std::size_t size() const noexcept;

            /// Remove all references to actions.
            void clear() noexcept;

            /// Evaluate the sum of actions for given auxilliary field phi.
            std::complex<double> eval(const CDVector &phi) const override;

            /// Calculate sum of forces for given auxilliary field phi.
            CDVector force(const CDVector &phi) const override;

        private:
            std::vector<Action*> _subActions;  ///< Stores individual summands.
        };
    }
}
#endif  // ndef ACTION_SUM_ACTION_HPP
