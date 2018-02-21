#ifndef PARDISO_HPP
#define PARDISO_HPP

/** \file
 * \brief Wrapper around PARDISO sparse solver.
 */

#include <iostream>
// #include <array>
#include <type_traits>
#include <memory>
#include "math.hpp"
#include "hubbardFermiMatrix.hpp"

/// Prototype for init function in PARDISO library.
extern "C" void pardisoinit(void *pt[64], int *mtype, int *solver, int iparm[64],
                            double dparm[64], int *error);

/// Prototype for solver execution function in PARDISO library.
extern "C" void pardiso(void *pt[64], int *maxfct, int *mnum, int *mtype,
                        int *phase, int *n, void *a, int ia[], int ja[],
                        int perm[], int *nrhs, int iparm[64], int *msglvl,
                        void *b, void *x, int *error, double dparm[64]);

/// Wrapper around PARDISO sparse solver.
namespace Pardiso {
    /// Solver kind used by PARDISO.
    enum class Solver {
        DIRECT = 0,    ///< Sparse direct solver.
        ITERATIVE = 1  ///< Multi-recursive iterative solver.
    };

    /// Matrix type for PARDISO excluding datatype.
    /**
     * Only encodes the symmetry type of matrices. The datatype is determined based
     * on the template parameter of Pardiso::State.
     *
     * Note that `SYM_HERM_INDEF`, `DIAGONAL`, and `BUNCH_KAUF` all represent the
     * same matrix type in PARDISO; they can be used interchangeably.
     */
    enum class MType {
        STRUCT_SYM,       ///< Structurally symmetric.
        SYM_HERM_POS_DEF, ///< Symmetric or hermitian positive definite.
        SYM_HERM_INDEF,   ///< Symmetric or hermitian undefinite.
        DIAGONAL,         ///< Diagonal.
        BUNCH_KAUF,       ///< Bunch-Kaufman pivoting.
        COMPL_SYM,        ///< Complex symmetric (not allowed with real matrices).
        NON_SYM           ///< Nonsymmetric.
    };

    /// A single PARDISO phase.
    /**
     * Cleaning up memory is done implicitly by Pardiso::State.
     *
     * \see pardisoPhase to build the combination of start / end phase.
     */
    enum class Phase {
        ANALYSIS = 1,      ///< Analysis.
        FACTORIZATION = 2, ///< Numerical factorization.
        SOLVE = 3,         ///< Solve / iterative refinement.
        SEL_INV = -22      ///< Selected inversion.
    };

    /// Construct phase parameter for calls to PARDISO from a start and end phase.
    inline int pardisoPhase(const Phase start, const Phase end) noexcept {
        if (start == Phase::SEL_INV)
            return static_cast<int>(start);

        return 10*static_cast<int>(start) + static_cast<int>(end);
    }

    /// Indices into integer parameters iparm of PARDISO.
    enum class IParm {
        USE_DEFAULT = 0,  ///< Fill iparm with default values (only for `pardisoinit`).
        NUM_PROC = 2      ///< Number of OpenMP threads.
    };

    /// Indices into double parameters dparm of PARDISO.
    enum class DParm {
        RESIDUAL = 33  ///< Relative residual after Krylov-Subspace convergence.
    };

    /// Check PARDISO error flag and throw exception if an error occured.
    /**
     * \param error PARDISO error code.
     * \throws `std::runtime_error` if `error != 0`. The exception message contains a brief
     *         description of the error.
     */
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


    ///
    /**
     * No copying because it references PARDISO's internal state.
     *
     * \tparam T Element type of matrices and vectors used by this solver.
     *           Must be one either `double` or `std::complex<double>`.
     */
    template <typename T>
    struct State {
        static_assert(std::is_same<T, double>::value
                      || std::is_same<T, std::complex<double>>::value,
                      "PARDISO can only handle double and std::complex<double>.");

        using elementType = T;  ///< Type of matrix elements.

        State(const Solver solver, const MType mtype = MType::NON_SYM,
              const int messageLevel=0)
            : _msglvl{messageLevel}, _mtype{matrixType(mtype)}, _ownsMemory{false} {

            int error;
            int slvr = static_cast<int>(solver);
            (*this)[IParm::USE_DEFAULT] = 0;  // fill in default values
            (*this)[IParm::NUM_PROC] = 1;
            pardisoinit(_statePtr.get(), &_mtype, &slvr,
                        _iparm.get(), _dparm.get(), &error);
            handleError(error);
        };
        
        State(const State &other) = delete;
        State &operator=(const State &other) = delete;

        State(State &&other) noexcept
            : _statePtr{std::move(other._statePtr)},
              _iparm{std::move(other._iparm)},
              _dparm{std::move(other._dparm)},
              _msglvl{other._msglvl},
              _mtype{other._mtype},
              _ownsMemory{std::exchange(other._ownsMemory, false)} { }

        State &operator=(State &&other) noexcept {
            _statePtr = std::move(other._statePtr);
            _iparm = std::move(other._iparm);
            _dparm = std::move(other._dparm);
            _msglvl = other._msglvl;
            _mtype = other._mtype;
            _ownsMemory = std::exchange(other._ownsMemory, false);
            return *this;
        }

        ~State() {
            clear();
        }

        /// Free all memory allocated by PARDISO; is called by destructor.
        void clear() {
            if (_ownsMemory) {
                int error;
                int phase = -1;
                pardiso(_statePtr.get(), nullptr, nullptr, nullptr,
                        &phase, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr, _iparm.get() , &_msglvl, nullptr,
                        nullptr, &error, _dparm.get());
                handleError(error);
                _ownsMemory = false;
            }            
        }


        /// Access an integer parameter.
        int &operator[](const IParm ip) noexcept {
            return _iparm[static_cast<std::size_t>(ip)];
        }
        /// Access an integer parameter.
        const int &operator[](const IParm ip) const noexcept {
            return _iparm[static_cast<std::size_t>(ip)];
        }

        /// Return pointer to array of integer parameters.
        int *iparm() noexcept {
            return _iparm.get();
        }
        /// Return pointer to array of integer parameters.
        const int *iparm() const noexcept {
            return _iparm.get();
        }

        /// Access a double parameter.
        double &operator[](const DParm dp) noexcept {
            return _iparm[static_cast<std::size_t>(dp)];
        }
        /// Access a double parameter.
        const double &operator[](const DParm dp) const noexcept {
            return _iparm[static_cast<std::size_t>(dp)];
        }

        /// Return pointer to array of double parameters.
        int *dparm() noexcept {
            return _dparm.get();
        }
        /// Return pointer to array of double parameters.
        const int *dparm() const noexcept {
            return _dparm.get();
        }


        /// Perform sparse solve by calling `pardiso`.
        /**
         * Solves a*x = b for x.<BR>
         * Low level interface to PARDISO. Matrix must be specified in CSR3 format
         * (three array variation of CSR format). Memory for the output must be allocated
         * by caller.
         *
         * \param n Number of equations (size of x, b).
         * \param a Non zero elements of matrix. Size depends on matrix.
         * \param ia Index to beginning to each row (1 based). Size: `n+1`.
         * \param ja Column indices for each element in `a`. Size depends on matrix.
         * \param b Right hand side vector. Size: `n`.
         * \param x Solution vector. Size: `n`.
         * \param startPhase Phase at which to start computation.
         * \param endPhase Phase at which to end computation.
         *
         * \throws std::runtime_error if PARDISO reports an error.
         */
        void operator()(int n,
                        elementType * const a, int * const ia, int * const ja,
                        elementType * const b, elementType * const x,
                        const Phase startPhase, const Phase endPhase=Phase::SOLVE) {

            int maxfct = 1, mnum = 1;
            int phase = pardisoPhase(startPhase, endPhase);
            int nrhs = 1;
            int error;
            
            pardiso(_statePtr.get(), &maxfct, &mnum, &_mtype, &phase,
                    &n, a, ia, ja, nullptr, &nrhs, _iparm.get(),
                    &_msglvl, b, x, &error, _dparm.get());
            handleError(error);

            _ownsMemory = true;
        }

        /// Perform sparse solve by calling `pardiso`.
        /**
         * Solves a*x = b for x.<BR>
         * Thin wrapper for `std::vector` around overload for arrays.
         * Matrix must be specified in CSR3 format (three array variation of CSR format).
         * Size n is derived from right hand side b.
         *
         * \param a Non zero elements of matrix. Size depends on matrix.
         * \param ia Index to beginning to each row (1 based). Size: `n+1`.
         * \param ja Column indices for each element in `a`. Size depends on matrix.
         * \param b Right hand side vector. Size: `n`.
         * \param startPhase Phase at which to start computation.
         * \param endPhase Phase at which to end computation.
         *
         * \return Solution vector x.
         *
         * \throws std::runtime_error if PARDISO reports an error.
         */
        std::vector<elementType> operator()(std::vector<elementType> &a,
                                            std::vector<int> &ia,
                                            std::vector<int> ja,
                                            std::vector<elementType> &b,
                                            const Phase startPhase,
                                            const Phase endPhase=Phase::SOLVE) {
            std::vector<elementType> x(b.size());
            (*this)(b.size(), &a[0], &ia[0], &ja[0], &b[0], &x[0], startPhase, endPhase);
            return x;
        }
        
    private:
        /// Pointer to the internal state of PARDISO.
        std::unique_ptr<void*[]> _statePtr = std::make_unique<void*[]>(64);
        /// Array of integer parameters.
        std::unique_ptr<int[]> _iparm = std::make_unique<int[]>(64);
        /// Array of double parameters.
        std::unique_ptr<double[]> _dparm = std::make_unique<double[]>(64);

        int _msglvl;      ///< Message level (0 or 1).
        int _mtype;       ///< PARDISO matrix type.
        bool _ownsMemory; ///< `true` if PARDISO allocated memory that needs to be freed.

        /// Return the full matrix type based on a symmetry type and datatype of this class.
        /**
         * \param mtype Determines the symmetry type of the matrix. The data type
         *        (real / complex) is determined from `State::elementType`.
         * \throws std::runtime_error if calling with `mtype == MType::COMPL_SYM` and
         *         `State::elementType == double`.
         * \returns Integer encoding PARDISO matrix type.
         */
        int matrixType(const MType mtype) const {
            if (std::is_same<elementType, double>::value) {
                switch (mtype) {
                case MType::NON_SYM:
                    return 11;
                case MType::STRUCT_SYM:
                    return 1;
                case MType::SYM_HERM_POS_DEF:
                    return 2;
                case MType::SYM_HERM_INDEF:
                case MType::DIAGONAL:
                case MType::BUNCH_KAUF:
                    return -2;
                case MType::COMPL_SYM:
                    throw std::runtime_error("Cannot use Pardiso::Mtype::COMPL_SYM with real matrix.");
                }
            } else if (std::is_same<elementType, std::complex<double>>::value) {
                switch (mtype) {
                case MType::NON_SYM:
                    return 13;
                case MType::STRUCT_SYM:
                    return 3;
                case MType::SYM_HERM_POS_DEF:
                    return 4;
                case MType::SYM_HERM_INDEF:
                case MType::DIAGONAL:
                case MType::BUNCH_KAUF:
                    return -4;
                case MType::COMPL_SYM:
                    return 6;
                }                
            }
        }
    };
}

#endif  // ndef PARDISO_HPP
