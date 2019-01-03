#include "hamiltonian.hpp"

namespace isle {
    namespace action {
        Action *Hamiltonian::add(Action *const action, const bool takeOwnership) {
            _actions.emplace_back(action, takeOwnership);
            return _actions.back().get();
        }

        Action *Hamiltonian::add(UnObHybridPtr<Action> &&action) {
            _actions.emplace_back(std::move(action));
            return _actions.back().get();
        }

        Action *Hamiltonian::operator[](std::size_t idx) noexcept(ndebug) {
#ifndef NDEBUG
            if (idx >= _actions.size())
                throw std::out_of_range("Index of action is out of range");
#endif
            return _actions[idx].get();
        }

        const Action *Hamiltonian::operator[](std::size_t idx) const noexcept(ndebug) {
#ifndef NDEBUG
            if (idx >= _actions.size())
                throw std::out_of_range("Index of action is out of range");
#endif
            return _actions[idx].get();
        }

        std::size_t Hamiltonian::size() const noexcept {
            return _actions.size();
        }

        void Hamiltonian::clear() noexcept {
            _actions.clear();
        }

        std::complex<double> Hamiltonian::eval(const Vector<std::complex<double>> &phi,
                                               const Vector<std::complex<double>> &pi) const {
            std::complex<double> res = 0;
            for (auto &act : _actions)
                res += act->eval(phi);
            return addMomentum(pi, res);
        }

        Vector<std::complex<double>> Hamiltonian::force(const Vector<std::complex<double>> &phi) const {
            Vector<std::complex<double>> res(phi.size(), 0);
            for (auto &act : _actions)
                res += act->force(phi);
            return res;
        }

        std::complex<double> Hamiltonian::addMomentum(const Vector<std::complex<double>> &pi,
                                                      const std::complex<double> action) const {
            return action + (blaze::conj(pi), pi)/2.;
        }

        std::complex<double> Hamiltonian::stripMomentum(const Vector<std::complex<double>> &pi,
                                                        const std::complex<double> action) const {
            return action - (blaze::conj(pi), pi)/2.;
        }
    }
}
