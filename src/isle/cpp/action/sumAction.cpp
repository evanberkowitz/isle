#include "sumAction.hpp"

namespace isle {
    namespace action {
        Action *SumAction::add(Action *const action) {
            _subActions.emplace_back(action);
            _trajHandles.emplace_back(action->beginTrajectory());
            return _subActions.back();
        }

        Action *SumAction::operator[](std::size_t idx) {
            if (idx >= _subActions.size())
                throw std::out_of_range("Index of action is out of range");
            return _subActions[idx];
        }

        const Action *SumAction::operator[](std::size_t idx) const {
            if (idx >= _subActions.size())
                throw std::out_of_range("Index of action is out of range");
            return _subActions[idx];
        }

        std::size_t SumAction::size() const noexcept {
            return _subActions.size();
        }

        void SumAction::clear() noexcept {
            _subActions.clear();
            _trajHandles.clear();
        }

        namespace {
            std::complex<double> sumActionEval(const CDVector &phi,
                                               std::vector<std::unique_ptr<BaseTrajectoryHandle>> const &handles) {
                std::complex<double> res = 0;
                for (auto &handle : handles)
                    res += handle->eval(phi);
                return res;
            }
        }

        std::complex<double> SumAction::eval(const CDVector &phi) const {
            return sumActionEval(phi, _trajHandles);
        }

        namespace {
            CDVector sumActionForce(const CDVector &phi,
                                    std::vector<std::unique_ptr<BaseTrajectoryHandle>> const &handles) {
                CDVector res(phi.size(), 0);
                for (auto &handle : handles)
                    res += handle->force(phi);
                return res;
            }
        }

        CDVector SumAction::force(const CDVector &phi) const {
            return sumActionForce(phi, _trajHandles);
        }


        /*
         *  -------------  TrajectoryHandle<SumAction>  -------------
         */


        TrajectoryHandle<SumAction>::TrajectoryHandle(const SumAction &sumAction)
        {
            _subHandles.reserve(sumAction.size());
            for (std::size_t i = 0; i < sumAction.size(); ++i) {
                _subHandles.emplace_back(sumAction[i]->beginTrajectory());
            }
        }

        std::complex<double> TrajectoryHandle<SumAction>::eval(const Vector<std::complex<double>> &phi) const {
            return sumActionEval(phi, _subHandles);
        }

        Vector<std::complex<double>> TrajectoryHandle<SumAction>::force(const Vector<std::complex<double>> &phi) const {
            return sumActionForce(phi, _subHandles);
        }

    }
}
