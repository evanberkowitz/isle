/** \file
 * \brief Collections of actions plus momentum term.
 */

#ifndef HAMILTONIAN_HPP
#define HAMILTONIAN_HPP

#include <vector>
#include <memory>
#include <initializer_list>

#include "core.hpp"
#include "action.hpp"

namespace cnxx {
    /// Collection of actions.
    /**
     * Stores arbitrary instances of derived types of Action.
     * Calling Hamiltonian::eval() evaluates all actions and adds the momentum term:
     \f[
     \texttt{Hamiltonian::eval(phi, pi)} = \frac{\texttt{pi}^2}{2}
                 + \sum_{S\in \mathrm{actions}}\,S(\texttt{phi})
     \f]
     * and similarily for Hamiltonian::force().
     */
    class Hamiltonian {
    public:
        Hamiltonian() = default;
        Hamiltonian(const Hamiltonian &other) = delete;
        Hamiltonian &operator=(const Hamiltonian &other) = delete;
        Hamiltonian(Hamiltonian &&other) = default;
        Hamiltonian &operator=(Hamiltonian &&other) = default;
        ~Hamiltonian() = default;

        /// Add an action to the collection.
        /**
         * \param action Pointer to the action to store. <B>%Hamiltonian assumes ownership
         *               of the object, i.e. it manages the actions lifetime.</B>
         * \returns `action`.
         */
        Action *add(Action *action);

        /// Construct a new action in place.
        /**
         * \tparam T Type of the action to construct.
         * \tparam Args Consturctor parameters of T.
         * \param args Parameters to pass to the constructor of T.
         * \returns A pointer to the newly created action. Ownership lies with Hamiltonian.
         */
        template <typename T, typename... Args>
        Action *emplace(Args... args) {
            _actions.emplace_back(new T{args...});
            return _actions.back().get();
        }

        /// Access an action via its index.
        Action *operator[](std::size_t idx) noexcept(ndebug);
        /// Access an action via its index.
        const Action *operator[](std::size_t idx) const noexcept(ndebug);

        /// Return the number of actions currently stored.
        std::size_t size() const noexcept;

        /// Remove all actions.
        void clear() noexcept;

        /// Evaluate the %Hamiltonian for given auxilliary field phi and momentum pi.
        /**
         * Evaluates all actions and adds the momentum term.
         */
        std::complex<double> eval(const Vector<std::complex<double>> &phi,
                                  const Vector<std::complex<double>> &pi);

        /// Calculate force for given auxilliary field phi and momentum pi.
        Vector<std::complex<double>> force(const Vector<std::complex<double>> &phi);

    private:
        std::vector<std::unique_ptr<Action>> _actions;  ///< Stores actions to evaluate.
    };
}

#endif  // ndef HAMILTONIAN_HPP
