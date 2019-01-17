#include "sumAction.hpp"

namespace isle {
    namespace action {
        Action *SumAction::add(Action *const action, const bool takeOwnership) {
            _subActions.emplace_back(action, takeOwnership);
            return _subActions.back().get();
        }

        Action *SumAction::add(UnObHybridPtr<Action> &&action) {
            _subActions.emplace_back(std::move(action));
            return _subActions.back().get();
        }

        Action *SumAction::operator[](std::size_t idx) noexcept(ndebug) {
#ifndef NDEBUG
            if (idx >= _subActions.size())
                throw std::out_of_range("Index of action is out of range");
#endif
            return _subActions[idx].get();
        }

        const Action *SumAction::operator[](std::size_t idx) const noexcept(ndebug) {
#ifndef NDEBUG
            if (idx >= _subActions.size())
                throw std::out_of_range("Index of action is out of range");
#endif
            return _subActions[idx].get();
        }

        std::size_t SumAction::size() const noexcept {
            return _subActions.size();
        }

        void SumAction::clear() noexcept {
            _subActions.clear();
        }

        std::complex<double> SumAction::eval(const CDVector &phi) const {
            std::complex<double> res = 0;
            for (auto &act : _subActions)
                res += act->eval(phi);
            return res;
        }

        CDVector SumAction::force(const CDVector &phi) const {
            CDVector res(phi.size(), 0);
            for (auto &act : _subActions)
                res += act->force(phi);
            return res;
        }
    }
}
