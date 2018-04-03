#include "hamiltonian.hpp"

#include <iostream>

namespace cnxx {
    Action *Hamiltonian::add(Action *const action) {
        _actions.emplace_back(action);
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
                                           const Vector<std::complex<double>> &pi) {
        std::complex<double> res = blaze::sqrNorm(pi)/2.;
        for (auto &act : _actions)
            res += act->eval(phi);
        return res;
    }

    Vector<std::complex<double>> Hamiltonian::force(const Vector<std::complex<double>> &phi) {
        Vector<std::complex<double>> res(phi.size(), 0);
        for (auto &act : _actions)
            res += act->force(phi);
        return res;
    }
}
