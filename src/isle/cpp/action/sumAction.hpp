/** \file
 * \brief Sum of other actions.
 */

#ifndef ACTION_SUM_ACTION_HPP
#define ACTION_SUM_ACTION_HPP

#include <vector>

#include "../core.hpp"
#include "action.hpp"
#include "../memory.hpp"

namespace isle {
    namespace action {
        /// Sum of several actions.
        /**
         * Stores arbitrary instances of derived types of Action.
         * Calling SumAction::eval() evaluates all actions and adds up the results;
         * similarily for SumAction::force().
         */
        class SumAction {
        public:
            SumAction() = default;
            SumAction(const SumAction &other) = delete;
            SumAction &operator=(const SumAction &other) = delete;
            SumAction(SumAction &&other) noexcept = default;
            SumAction &operator=(SumAction &&other) noexcept = default;
            ~SumAction() noexcept = default;

            /// Add an action to the collection.
            /**
             * \param action Pointer to the action to store.
             * \param takeOwnership If `true` %SumAction takes ownership of the
             *                      action and deletes it in its destructor.
             *                      If `false` ownership lies with someone else and the user
             *                      is responsible for cleaning up.
             * \returns `action`.
             */
            Action *add(Action *action, bool takeOwnership);

            /// Add an action to the collection.
            /**
             * \param action Pointer to the action to store. Onwership is transferred to
             *               %SumAction if the UnObHybridPtr owns the action.
             * \returns `action`.
             */
            Action *add(UnObHybridPtr<Action> &&action);

            /// Construct a new action in place.
            /**
             * \tparam T Type of the action to construct.
             * \tparam Args Consturctor parameters of T.
             * \param args Parameters to pass to the constructor of T.
             * \returns A pointer to the newly created action. Ownership lies with SumAction.
             */
            template <typename T, typename... Args>
            Action *emplace(Args... args) {
                _subActions.emplace_back(new T{std::forward<Args>(args)...}, true);
                return _subActions.back().get();
            }

            /// Access an action via its index.
            Action *operator[](std::size_t idx) noexcept(ndebug);
            /// Access an action via its index.
            const Action *operator[](std::size_t idx) const noexcept(ndebug);

            /// Return the number of actions currently stored.
            std::size_t size() const noexcept;

            /// Remove all actions.
            void clear() noexcept;

            /// Evaluate the sum of actions for given auxilliary field phi.
            std::complex<double> eval(const CDVector &phi) const;

            /// Calculate sum of forces for given auxilliary field phi.
            CDVector force(const CDVector &phi) const ;

        private:
            std::vector<UnObHybridPtr<Action>> _subActions;  ///< Stores individual summands.
        };
    }
}
#endif  // ndef ACTION_SUM_ACTION_HPP
