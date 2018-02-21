#ifndef PARDISO_HPP
#define PARDISO_HPP

#include <iostream>
#include <type_traits>
#include <memory>

extern "C" void pardisoinit(void *pt[64], int *mtype, int *solver, int iparm[64],
                            double dparm[64], int *error);

extern "C" void pardiso(void *pt[64], int *maxfct, int *mnum, int *mtype,
                        int *phase, int *n, void *a, int ia[], int ja[],
                        int perm[], int *nrhs, int iparm[64], int *msglvl,
                        void *b, void *x, int *error, double dparm[64]);

namespace Pardiso {
    enum class Solver {
        DIRECT = 0,
        ITERATIVE = 1
    };

    enum class Phase {
        ANALYSIS = 1,
        FACTORIZATION = 2,
        SOLVE = 3,
        SEL_INV = -22
    };

    enum class IParm {
        USE_DEFAULT = 0,
        NUM_PROC = 2
    };

    enum class DParm {
        RESIDUAL = 33
    };
    
    inline void handleError(const int error) {
        switch (error) {
        case 0:
            return;
        case -1:
            throw std::runtime_error("PARDISO Error -1: Input inconsistent.");
        case -2:
            throw std::runtime_error("PARDISO Error -2: Not enough memory");
        case -3:
            throw std::runtime_error("PARDISO Error -3: Reordering problem");
        case -4:
            throw std::runtime_error("PARDISO Error -4: Zero pivot, numerical fact. or iterative refinement problem");
        case -5:
            throw std::runtime_error("PARDISO Error -5: Unclassified (internal) error");
        case -6:
            throw std::runtime_error("PARDISO Error -6: Preordering failed");
        case -7:
            throw std::runtime_error("PARDISO Error -7: Diagonal matrix problem");
        case -8:
            throw std::runtime_error("PARDISO Error -8: 32-bit integer overflow problem");
        case -10:
            throw std::runtime_error("PARDISO Error -10: No license file pardiso.lic found");
        case -11:
            throw std::runtime_error("PARDISO Error -11: License is expired");
        case -12:
            throw std::runtime_error("PARDISO Error -12: Wrong username or hostname");
        case -100:
            throw std::runtime_error("PARDISO Error -100: Reached maximum number of Krylov-subspace iteration");
        case -101:
            throw std::runtime_error("PARDISO Error -101: No sufficient convergence in Krylov-subspace iteration within 25 iterations");
        case -102:
            throw std::runtime_error("PARDISO Error -102: Error in Krylov-subspace iteration");
        case -103:
            throw std::runtime_error("PARDISO Error -103: Break-down in Krylov-subspace iteration");
        default:
            throw std::runtime_error("Unknown PARDISO Error");
        }
    }

    inline int pardisoPhase(const Phase start, const Phase end) noexcept {
        if (start == Phase::SEL_INV)
            return static_cast<int>(start);

        return 10*static_cast<int>(start) + static_cast<int>(end);
    }
    
    template <typename T>
    struct State {
        static_assert(std::is_same<T, double>::value
                      || std::is_same<T, std::complex<double>>::value,
                      "PARDISO can only handle double and std::complex<double>.");

        using valueType = T;

        State(const Solver solver, const int messageLevel=0)
            : msglvl{messageLevel}, ownsMemory{false} {

            int mt = mtype();
            int error;
            int slvr = static_cast<int>(solver);
            iparm[0] = 0;  // use default values
            pardisoinit(statePtr.get(), &mt, &slvr, iparm.get(), dparm.get(), &error);
            handleError(error);
        };

        State(const State &other) = delete;
        State &operator=(const State &other) = delete;

        State(State &&other) noexcept
            : statePtr{std::move(other.statePtr)},
              iparm{std::move(other.iparm)},
              dparm{std::move(other.dparm)},
              msglvl{other.msglvl},
              ownsMemory{std::exchange(other.ownsMemory, false)} { }

        State &operator=(State &&other) noexcept {
            statePtr = std::move(other.statePtr);
            iparm = std::move(other.iparm);
            dparm = std::move(other.dparm);
            msglvl = other.msglvl;
            ownsMemory = std::exchange(other.ownsMemory, false);
            return *this;
        }

        ~State() {
            clear();
        }


        void clear() {
            if (ownsMemory) {
                int error;
                int phase = -1;
                pardiso(statePtr.get(), nullptr, nullptr, nullptr,
                        &phase, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr, iparm.get() , &msglvl, nullptr,
                        nullptr, &error, dparm.get());
                handleError(error);
                ownsMemory = false;
            }            
        }


        int &operator[](const IParm ip) noexcept {
            return iparm[static_cast<std::size_t>(ip)];
        }

        const int &operator[](const IParm ip) const noexcept {
            return iparm[static_cast<std::size_t>(ip)];
        }

        double &operator[](const DParm dp) noexcept {
            return iparm[static_cast<std::size_t>(dp)];
        }

        const double &operator[](const DParm dp) const noexcept {
            return iparm[static_cast<std::size_t>(dp)];
        }


        void operator()(int n,
                        T * const a, int * const ia, int * const ja,
                        T * const b, T * const x,
                        const Phase startPhase, const Phase endPhase=Phase::SOLVE) {

            int maxfct = 1, mnum = 1;
            int mt = mtype();
            int phase = pardisoPhase(startPhase, endPhase);
            int nrhs = 1;
            int error;
            
            pardiso(statePtr.get(), &maxfct, &mnum, &mt, &phase,
                    &n, a, ia, ja, nullptr, &nrhs, iparm.get(),
                    &msglvl, b, x, &error, dparm.get());
            handleError(error);

            ownsMemory = true;
        }
        
    private:
        std::unique_ptr<void*[]> statePtr = std::make_unique<void*[]>(64);
        std::unique_ptr<int[]> iparm = std::make_unique<int[]>(64);
        std::unique_ptr<double[]> dparm = std::make_unique<double[]>(64);

        int msglvl;
        bool ownsMemory;
        
        int mtype() const noexcept {
            if (std::is_same<T, double>::value)
                return 11;
            else if (std::is_same<T, std::complex<double>>::value)
                return 13;
        }
    };
}

#endif  // ndef PARDISO_HPP
