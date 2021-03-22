#include "hubbardFermiMatrixExp.hpp"

#include <memory>
#include <limits>
#include <cmath>

#include "profile.hpp"
#include "logging/logging.hpp"

namespace isle {
    // wrapper declariation for the GPU port
#ifdef USE_CUDA
    void F_wrapper(std::complex<double> *, std::size_t ,const std::complex<double> *,const double *,const std::size_t, const std::size_t,const Species, const bool);
#endif

    namespace {
        /// Resize a square matrix, throws away old elements.
        template <typename MT>
        void resizeMatrix(MT &mat, const std::size_t target) {
#ifndef NDEBUG
            if (mat.rows() != mat.columns())
                throw std::invalid_argument("Matrix is not square.");
#endif
            if (mat.rows() != target)
                mat.resize(target, target, false);
        }

        auto computeExponential(const DSparseMatrix &kappa,
                                const double mu,
                                const std::int8_t sigmaKappa,
                                const Species species,
                                const bool inv) {
            switch (species) {
            case Species::PARTICLE:
                if (inv) {
                    return expmSym(-kappa + mu*IdMatrix<double>(kappa.rows()));
                } else {
                    return expmSym(kappa - mu*IdMatrix<double>(kappa.rows()));
                }
            case Species::HOLE:
                if (inv) {
                    return expmSym(-sigmaKappa*kappa - mu*IdMatrix<double>(kappa.rows()));
                } else {
                    return expmSym(sigmaKappa*kappa + mu*IdMatrix<double>(kappa.rows()));
                }
            }
            // Strictly speaking impossible to reach but gcc complains.
            throw std::invalid_argument("Unknown species");
        }
    }

/*
 * -------------------------- HubbardFermiMatrixExp --------------------------
 */

    HubbardFermiMatrixExp::HubbardFermiMatrixExp(const DSparseMatrix &kappaTilde,
                                                 const double muTilde,
                                                 const std::int8_t sigmaKappa)
        : _kappa{kappaTilde}, _mu{muTilde}, _sigmaKappa{sigmaKappa},
          _expKappap{computeExponential(kappaTilde, muTilde, sigmaKappa, Species::PARTICLE, false)},
          _expKappapInv{computeExponential(kappaTilde, muTilde, sigmaKappa, Species::PARTICLE, true)},
          _expKappah{computeExponential(kappaTilde, muTilde, sigmaKappa, Species::HOLE, false)},
          _expKappahInv{computeExponential(kappaTilde, muTilde, sigmaKappa, Species::HOLE, true)},
          _logdetExpKappahInv{logdet(_expKappahInv)}
    {
        if (kappaTilde.rows() != kappaTilde.columns())
            throw std::invalid_argument("Hopping matrix is not square.");
        if (sigmaKappa != +1 && sigmaKappa != -1)
            getLogger("HubbardFermiMatrixExp").warning("sigmaKappa should be either -1 or +1.");
        if (sigmaKappa == +1 && !isBipartite(kappaTilde))
            getLogger("HubbardFermiMatrixExp").warning("sigmaKappa should be -1 because the lattice is not bipartite.");
    }

    HubbardFermiMatrixExp::HubbardFermiMatrixExp(const Lattice &lat,
                                                 const double beta,
                                                 const double muTilde,
                                                 const std::int8_t sigmaKappa)
        : HubbardFermiMatrixExp{lat.hopping()*beta/lat.nt(), muTilde, sigmaKappa} { }

    const DMatrix &HubbardFermiMatrixExp::expKappa(const Species species, const bool inv) const {
        switch (species) {
        case Species::PARTICLE:
            if (inv) {
                return _expKappapInv;
            } else {
                return _expKappap;
            }
        case Species::HOLE:
            if (inv) {
                return _expKappahInv;
            } else {
                return _expKappah;
            }
        }
        // Strictly speaking impossible to reach but gcc complains.
        throw std::invalid_argument("Unknown species");
    }

    std::complex<double> HubbardFermiMatrixExp::logdetExpKappa(const Species species,
                                                               const bool inv) const {
        if (species == Species::HOLE && inv) {
            return _logdetExpKappahInv;
        }
        throw std::runtime_error("logdetExpKappa is only implemented for holes and inv=true");
    }

    void HubbardFermiMatrixExp::K(DSparseMatrix &k, const Species UNUSED(species)) const {
        k = IdMatrix<double>(nx());
    }

    IdMatrix<double> HubbardFermiMatrixExp::K(const Species UNUSED(species)) const {
        return IdMatrix<double>(nx());
    }

    DMatrix HubbardFermiMatrixExp::Kinv(const Species UNUSED(species)) const {
        return IdMatrix<double>(nx());
    }

    std::complex<double> HubbardFermiMatrixExp::logdetKinv(Species UNUSED(species)) const {
        return 0;
    }

    void HubbardFermiMatrixExp::F(CDMatrix &f,
                                  const std::size_t tp, const CDVector &phi,
                                  const Species species, const bool inv) const {
        ISLE_PROFILE_NVTX_RANGE(species == Species::PARTICLE
                                ? "HubbardFermiMatrixExp::F(particle)"
                                : "HubbardFermiMatrixExp::F(hole)");
        const std::size_t NX = nx();
        const std::size_t NT = getNt(phi, NX);
        resizeMatrix(f, NX);

        #ifdef USE_CUDA

        F_wrapper(f.data(),tp,phi.data(),expKappa(species,inv).data(), NX, NT,species,inv);

        #else // USE_CUDA

        const std::size_t tm1 = tp==0 ? NT-1 : tp-1;  // t' - 1

        // the sign in the exponential of phi
        auto const sign = ((species == Species::PARTICLE && !inv)
                           || (species == Species::HOLE && inv))
            ? std::complex<double>{0.0, +1.0}
            : std::complex<double>{0.0, -1.0};

        if (inv)
            // f = e^phi * e^kappa  (up to signs in exponents)
            f = expand(exp(sign*spacevec(phi, tm1, NX)), NX)
                % expKappa(species, inv);
        else
            // f = e^kappa * e^phi  (up to signs in exponents)
            f = expKappa(species, inv)
                % expand(trans(exp(sign*spacevec(phi, tm1, NX))), NX);

        #endif // USE_CUDA
    }

    CDMatrix HubbardFermiMatrixExp::F(const std::size_t tp, const CDVector &phi,
                                      const Species species, const bool inv) const {
        CDMatrix f;
        F(f, tp, phi, species, inv);
        return f;
    }

    void HubbardFermiMatrixExp::M(CDSparseMatrix &m,
                                  const CDVector &phi,
                                  const Species species) const {
        const std::size_t NX = nx();
        const std::size_t NT = getNt(phi, NX);
        m = IdMatrix<double>(NX*NT);

        // zeroth row w/ explicit boundary condition
        auto f = F(0, phi, species);
        spacemat(m, 0, NT-1, NX) = f;

        // other rows
        for (std::size_t tp = 1; tp < NT; ++tp) {
            F(f, tp, phi, species);
            spacemat(m, tp, tp-1, NX) = -f;
        }
    }

    CDSparseMatrix HubbardFermiMatrixExp::M(const CDVector &phi,
                                            const Species species) const {
        CDSparseMatrix m;
        M(m, phi, species);
        return m;
    }


    void HubbardFermiMatrixExp::P(DMatrix &P) const {
        P = this->P();
    }

    DMatrix HubbardFermiMatrixExp::P() const {
        if (_sigmaKappa == -1)  // the exponentials are ivnerses of each other
            return 2*IdMatrix<double>(nx());
        else
            return IdMatrix<double>(nx()) \
                + expKappa(Species::PARTICLE, false)*expKappa(Species::HOLE, false);
    }

    void HubbardFermiMatrixExp::Tplus(CDMatrix &T,
                                      const std::size_t tp,
                                      const CDVector &phi) const {
        const double antiPSign = tp==0 ? -1 : 1;   // encode anti-periodic BCs
        T = -antiPSign*F(tp, phi, Species::PARTICLE, false);
    }

    CDMatrix HubbardFermiMatrixExp::Tplus(const std::size_t tp,
                                          const CDVector &phi) const {
        CDMatrix T;
        Tplus(T, tp, phi);
        return T;
    }

    void HubbardFermiMatrixExp::Tminus(CDMatrix &T,
                                       const std::size_t tp,
                                       const CDVector &phi) const {
        const std::size_t NT = getNt(phi, nx());
        const double antiPSign = tp==NT-1 ? -1 : 1;   // encode anti-periodic BCs
        T = -antiPSign*blaze::trans(F(loopIdx(tp+1, NT), phi, Species::HOLE, false));
    }

    CDMatrix HubbardFermiMatrixExp::Tminus(const std::size_t tp,
                                           const CDVector &phi) const {
        CDMatrix T;
        Tminus(T, tp, phi);
        return T;
    }

    void HubbardFermiMatrixExp::Q(CDSparseMatrix &q, const CDVector &phi) const {
        const std::size_t NX = nx();
        const std::size_t NT = getNt(phi, NX);
        resizeMatrix(q, NX*NT);

        const auto p = P();
        CDMatrix aux; // for both T^+ and T^-
        for (std::size_t tp = 0; tp < NT; ++tp) {
            spacemat(q, tp, tp, NX) = p;

            Tplus(aux, tp, phi);
            spacemat(q, tp, (tp+NT-1)%NT, NX) += aux;

            Tminus(aux, tp, phi);
            spacemat(q, tp, (tp+1)%NT, NX) += aux;
        }
    }

    CDSparseMatrix HubbardFermiMatrixExp::Q(const CDVector &phi) const {
        CDSparseMatrix q;
        Q(q, phi);
        return q;
    }


    void HubbardFermiMatrixExp::updateKappaTilde(const SparseMatrix<double> &kappaTilde) {
        _kappa = kappaTilde;
        _expKappap = computeExponential(kappaTilde, _mu, _sigmaKappa, Species::PARTICLE, false);
        _expKappapInv = computeExponential(kappaTilde, _mu, _sigmaKappa, Species::PARTICLE, true);
        _expKappah = computeExponential(kappaTilde, _mu, _sigmaKappa, Species::HOLE, false);
        _expKappahInv = computeExponential(kappaTilde, _mu, _sigmaKappa, Species::HOLE, true);
    }

    void HubbardFermiMatrixExp::updateMuTilde(const double muTilde) {
        _mu = muTilde;
        _expKappap = computeExponential(_kappa, _mu, _sigmaKappa, Species::PARTICLE, false);
        _expKappap = computeExponential(_kappa, _mu, _sigmaKappa, Species::PARTICLE, true);
        _expKappah = computeExponential(_kappa, _mu, _sigmaKappa, Species::HOLE, false);
        _expKappah = computeExponential(_kappa, _mu, _sigmaKappa, Species::HOLE, true);
    }

    const SparseMatrix<double> &HubbardFermiMatrixExp::kappaTilde() const noexcept {
        return _kappa;
    }

    double HubbardFermiMatrixExp::muTilde() const noexcept {
        return _mu;
    }

    std::int8_t HubbardFermiMatrixExp::sigmaKappa() const noexcept {
        return _sigmaKappa;
    }

    std::size_t HubbardFermiMatrixExp::nx() const noexcept {
        return _kappa.rows();
    }

/*
 * -------------------------- QLU --------------------------
 */

    HubbardFermiMatrixExp::QLU::QLU(const std::size_t nt) {
        dinv.reserve(nt);
        if (nt > 1) {
            u.reserve(nt-1);
            l.reserve(nt-1);
            if (nt > 2) {
                v.reserve(nt-2);
                h.reserve(nt-2);
            }
        }
    }

    bool HubbardFermiMatrixExp::QLU::isConsistent() const {
        const std::size_t nt = dinv.size();
        if (nt == 0)
            return false;
        else {
            // check this for any nt > 0
            if (u.size() != nt-1 || l.size() != nt-1)
                return false;
            // this check only works for nt > 1
            if (nt > 1 && (v.size() != nt-2 || h.size() != nt-2))
                return false;
        }
        return true;
    }

    CDMatrix HubbardFermiMatrixExp::QLU::reconstruct() const {
        const std::size_t nt = dinv.size();
        if (nt < 2)
            throw std::domain_error("Reconstruction of LU called with nt < 2");
#ifndef NDEBUG
        if (!isConsistent())
            throw std::runtime_error("Components of LU not initialized properly");
#endif

        const std::size_t nx = dinv[0].rows();
        std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(nx);// pivot indices for inversion
        CDMatrix aux;

        CDMatrix recon(nx*nt, nx*nt, 0);

        // zeroth row, all columns
        invert(aux=dinv[0], ipiv);
        spacemat(recon, 0, 0, nx) = aux;
        spacemat(recon, 0, 1, nx) = u[0];
        spacemat(recon, 0, nt-1, nx) = v[0];

        // rows 1 - nt-3 (all 'regular' ones), all columns
        for (std::size_t i = 1; i < nt-2; ++i) {
            invert(aux=dinv[i], ipiv);
            spacemat(recon, i, i, nx) = aux + l[i-1]*u[i-1];
            invert(aux=dinv[i-1], ipiv);
            spacemat(recon, i, i-1, nx) = l[i-1]*aux;
            spacemat(recon, i, i+1, nx) = u[i];
            spacemat(recon, i, nt-1, nx) = l[i-1]*v[i-1] + v[i];
        }

        // row nt-2, all columns
        invert(aux=dinv[nt-2], ipiv);
        spacemat(recon, nt-2, nt-2, nx) = aux + l[nt-3]*u[nt-3];
        invert(aux=dinv[nt-3], ipiv);
        spacemat(recon, nt-2, nt-3, nx) = l[nt-3]*aux;
        spacemat(recon, nt-2, nt-1, nx) = l[nt-3]*v[nt-3] + u[nt-2];

        // row nt-1, up to column nt-3
        invert(aux=dinv[0], ipiv);
        spacemat(recon, nt-1, 0, nx) = h[0]*aux;
        for (std::size_t i = 1; i < nt-2; ++i) {
            invert(aux=dinv[i], ipiv);
            spacemat(recon, nt-1, i, nx) = h[i-1]*u[i-1] + h[i]*aux;
        }

        // row nt-1, columns nt-2, nt-1
        invert(aux=dinv[nt-2], ipiv);
        spacemat(recon, nt-1, nt-2, nx) = h[nt-3]*u[nt-3] + l[nt-2]*aux;
        invert(aux=dinv[nt-1], ipiv);
        spacemat(recon, nt-1, nt-1, nx) = aux + l[nt-2]*u[nt-2];
        for (std::size_t i = 0; i < nt-2; ++i)
            spacemat(recon, nt-1, nt-1, nx) += h[i]*v[i];

        return recon;
    }

/*
 * -------------------------- free functions --------------------------
 */

    namespace {
        /// Special case LU decomposition of Q for nt == 1.
        HubbardFermiMatrixExp::QLU nt1QLU(const HubbardFermiMatrixExp &hfm,
                                          const CDVector &phi) {
            HubbardFermiMatrixExp::QLU lu{1};

            // construct d_0
            CDMatrix T = hfm.Tplus(0, phi);
            lu.dinv.emplace_back(hfm.P() + T);
            hfm.Tminus(T, 0, phi);
            lu.dinv[0] += T;

            // invert d_0
            std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(phi.size());
            invert(lu.dinv[0], ipiv);

            return lu;
        }

        /// Special case LU decomposition of Q for nt == 2.
        HubbardFermiMatrixExp::QLU nt2QLU(const HubbardFermiMatrixExp &hfm,
                                          const CDVector &phi) {
            const std::size_t nx = hfm.nx();
            constexpr std::size_t nt = 2;
            HubbardFermiMatrixExp::QLU lu{nt};

            const auto P = hfm.P();  // diagonal block P
            CDMatrix aux0, aux1; // T^+, T^-, and u
            std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(nx);// pivot indices for inversion

            // d_0
            lu.dinv.emplace_back(P);
            invert(lu.dinv[0], ipiv);

            // l_0
            hfm.Tplus(aux0, 1, phi);
            hfm.Tminus(aux1, 1, phi);
            lu.l.emplace_back((aux0+aux1)*lu.dinv[0]);

            // u_0
            hfm.Tplus(aux0, 0, phi);
            hfm.Tminus(aux1, 0, phi);
            aux0 += aux1;  // now aux0 = u_0
            lu.u.emplace_back(aux0);

            // d_1
            lu.dinv.emplace_back(P - lu.l[0]*aux0);
            invert(lu.dinv[1], ipiv);

            return lu;
        }

        /// General case LU decomposition of Q for nt > 2.
        HubbardFermiMatrixExp::QLU generalQLU(const HubbardFermiMatrixExp &hfm,
                                              const CDVector &phi) {
            const std::size_t nx = hfm.nx();
            const std::size_t nt = getNt(phi, nx);
            HubbardFermiMatrixExp::QLU lu{nt};

            const auto P = hfm.P();  // diagonal block P
            CDMatrix T;  // subdiagonal blocks T^+ and T^-

            CDMatrix u; // previous u, updated along the way
            // pivot indices for inversion
            std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(nx);

            // starting components of d, u, l
            lu.dinv.emplace_back(P);
            invert(lu.dinv.back(), ipiv);
            hfm.Tminus(u, 0, phi);   // now u = lu.u[0]
            lu.u.emplace_back(u);
            hfm.Tplus(T, 1, phi);
            lu.l.emplace_back(T*lu.dinv.back());

            // v, h
            hfm.Tplus(T, 0, phi);
            lu.v.emplace_back(T);
            hfm.Tminus(T, nt-1, phi);
            lu.h.emplace_back(T*lu.dinv.back());

            // iterate for i in [1, nt-3], 'regular' part of d, u, l, v, h
            for (std::size_t i = 1; i < nt-2; ++i) {
                // here, u = lu.u[i-1]
                lu.dinv.emplace_back(P - lu.l[i-1]*u);
                invert(lu.dinv.back(), ipiv);

                hfm.Tplus(T, i+1, phi);
                lu.l.emplace_back(T*lu.dinv[i]);
                lu.h.emplace_back(-lu.h[i-1]*u*lu.dinv[i]);
                lu.v.emplace_back(-lu.l[i-1]*lu.v[i-1]);

                hfm.Tminus(u, i, phi);
                lu.u.emplace_back(u);  // now u = lu.u[i]
            }
            // from now on u is lu.u[nt-3]

            // additional 'regular' step for d
            lu.dinv.emplace_back(P - lu.l[nt-3]*u);
            invert(lu.dinv.back(), ipiv);

            // final components of u, l
            hfm.Tminus(T, nt-2, phi);
            lu.u.emplace_back(T - lu.l[nt-3]*lu.v[nt-3]);
            hfm.Tplus(T, nt-1, phi);
            lu.l.emplace_back((T - lu.h[nt-3]*u)*lu.dinv[nt-2]);

            // final component of d
            lu.dinv.emplace_back(P - lu.l[nt-2]*lu.u[nt-2]);
            for (std::size_t i = 0; i < nt-2; ++i)
                lu.dinv[nt-1] -= lu.h[i]*lu.v[i];
            invert(lu.dinv.back(), ipiv);

            return lu;
        }
    }

    HubbardFermiMatrixExp::QLU getQLU(const HubbardFermiMatrixExp &hfm,
                                   const CDVector &phi) {
        switch (phi.size()/hfm.nx()) {
        case 1:
            return nt1QLU(hfm, phi);
        case 2:
            return nt2QLU(hfm, phi);
        default:
            return generalQLU(hfm, phi);
        }
    }

    Vector<std::complex<double>> solveQ(const HubbardFermiMatrixExp &hfm,
                                        const CDVector &phi,
                                        const Vector<std::complex<double>> &rhs) {
        return solveQ(getQLU(hfm, phi), rhs);
    }

    Vector<std::complex<double>> solveQ(const HubbardFermiMatrixExp::QLU &lu,
                                        const Vector<std::complex<double>> &rhs) {
        const std::size_t nt = lu.dinv.size();
#ifndef NDEBUG
        if (!lu.isConsistent())
            throw std::runtime_error("Components of LU not initialized properly");
        if (rhs.size() != lu.dinv[0].rows()*nt)
            throw std::runtime_error("Right hand side does not have correct size (spacetime vector)");
#endif

        const std::size_t nx = lu.dinv[0].rows();

        // solve L*y = rhs
        Vector<std::complex<double>> y(nt*nx);
        spacevec(y, 0, nx) = spacevec(rhs, 0, nx);
        for (std::size_t i = 1; i < nt-1; ++i)
            spacevec(y, i, nx) = spacevec(rhs, i, nx) - lu.l[i-1]*spacevec(y, i-1, nx);
        if (nt > 1) {
            spacevec(y, nt-1, nx) = spacevec(rhs, nt-1, nx) - lu.l[nt-2]*spacevec(y, nt-2, nx);
            for (std::size_t j = 0; j < nt-2; ++j)
                spacevec(y, nt-1, nx) -= lu.h[j]*spacevec(y, j, nx);
        }

        // solve U*x = y
        Vector<std::complex<double>> x(nt*nx);

        spacevec(x, nt-1, nx) = lu.dinv[nt-1]*spacevec(y, nt-1, nx);
        if (nt > 1) {
            spacevec(x, nt-2, nx) = lu.dinv[nt-2]*(spacevec(y, nt-2, nx)
                                                   - lu.u[nt-2]*spacevec(x, nt-1, nx));
            // iterate i in [nt-3, 0]
            for (std::size_t i = nt-3; i != static_cast<std::size_t>(-1); --i)
                spacevec(x, i, nx) = lu.dinv[i]*(spacevec(y, i, nx)
                                                 - lu.u[i]*spacevec(x, i+1, nx)
                                                 - lu.v[i]*spacevec(x, nt-1, nx));
        }

        return x;
    }

    std::complex<double> logdetQ(const HubbardFermiMatrixExp &hfm,
                                 const CDVector &phi) {
        auto lu = getQLU(hfm, phi);
        return ilogdetQ(lu);
    }

    std::complex<double> logdetQ(const HubbardFermiMatrixExp::QLU &lu) {
#ifndef NDEBUG
        if (!lu.isConsistent())
            throw std::runtime_error("Components of LU not initialized properly");
#endif

        std::complex<double> ldet;
        // calculate logdet of diagonal blocks
        for (const auto &dinv : lu.dinv)
            ldet -= logdet(dinv);
        return toFirstLogBranch(ldet);
    }

    std::complex<double> ilogdetQ(HubbardFermiMatrixExp::QLU &lu) {
#ifndef NDEBUG
        if (!lu.isConsistent())
            throw std::runtime_error("Components of LU not initialized properly");
#endif

        std::complex<double> ldet;
        // calculate logdet of diagonal blocks
        for (auto &dinv : lu.dinv)
            ldet -= ilogdet(dinv);
        return toFirstLogBranch(ldet);
    }

    namespace {
        // Use version log(det(1+hat{A})).
        std::complex<double> logdetM_p(const HubbardFermiMatrixExp &hfm,
                                       const CDVector &phi) {
            ISLE_PROFILE_NVTX_RANGE("logdetM_p(Exp)");
            const auto NX = hfm.nx();
            const auto NT = getNt(phi, NX);
            const auto species = Species::PARTICLE;

            // first factor F
            CDMatrix f;
            CDMatrix B = hfm.F(0, phi, species, false);
            // other factors
            for (std::size_t t = 1; t < NT; ++t) {
                hfm.F(f, t, phi, species, false);
                B = f*B;
            }

            B += IdMatrix<std::complex<double>>(NX);
            return toFirstLogBranch(ilogdet(B));
        }

        // Use version -i Phi - N_t log(det(e^{-sigmaKappa*kappa-mu})) + log(det(1+hat{A}^{-1})).
        std::complex<double> logdetM_h(const HubbardFermiMatrixExp &hfm,
                                       const CDVector &phi) {
            ISLE_PROFILE_NVTX_RANGE("logdetM_h(Exp)");
            const auto NX = hfm.nx();
            const auto NT = getNt(phi, NX);

            // build product of F^{-1}
            auto f = hfm.F(0, phi, Species::HOLE, true);
            CDMatrix aux = f;  // the matrix under the determinant
            for (std::size_t t = 1; t < NT; ++t) {
                hfm.F(f, t, phi, Species::HOLE, true);
                aux = aux*f;
            }
            aux += IdMatrix<std::complex<double>>(NX);

            // add Phi and return
            return toFirstLogBranch(-static_cast<double>(NT)*hfm.logdetExpKappa(Species::HOLE, true)
                                    - std::complex<double>{0.0, 1.0}*blaze::sum(phi)
                                    + ilogdet(aux));
        }
    }

    std::complex<double> logdetM(const HubbardFermiMatrixExp &hfm,
                                 const CDVector &phi, const Species species) {
        switch (species) {
        case Species::PARTICLE:
            return logdetM_p(hfm, phi);
        case Species::HOLE:
            return logdetM_h(hfm, phi);
        }
        // Strictly speaking impossible to reach but gcc complains.
        throw std::invalid_argument("Unknown species");
    }

    namespace {
#ifndef NDEBUG
        void verifyResultOfSolveM(const HubbardFermiMatrixExp &hfm,
                                  const CDVector &phi,
                                  const Species species,
                                  const CDMatrix &res,
                                  const CDMatrix &rhss) {
            const double diff = blaze::max(blaze::abs(hfm.M(phi, species) * blaze::trans(res)
                                                      - blaze::trans(rhss)));
            if (diff > 1e-8) {
                std::ostringstream oss;
                oss << "Check of result of solveM exceeds tolerance: " << diff << '\n';
                getLogger("HubbardFermiMatrixExp").warning(oss.str());
            }
        }
#endif
    }

    CDMatrix solveM(const HubbardFermiMatrixExp &hfm, const CDVector &phi,
                    const Species species, const CDMatrix &rhss) {
        ISLE_PROFILE_NVTX_RANGE(species == Species::PARTICLE
                                ? "solveM(Exp, particle)"
                                : "solveM(Exp, hole)");
        const std::size_t NX = hfm.nx();
        const std::size_t NT = getNt(phi, NX);
        const std::size_t NRHS = rhss.rows();
        // the results (vectors x in the end, z at intermediate stage)
        CDMatrix res(rhss.rows(), rhss.columns());

        // construct all partial A^{-1} and the complete one
        std::vector<CDMatrix> partialAinv;
        partialAinv.reserve(NT);
        partialAinv.emplace_back(hfm.F(0, phi, species, true));
        for (std::size_t t = 1; t < NT; ++t) {
            partialAinv.emplace_back(partialAinv[t-1]*hfm.F(t, phi, species, true));
        }

        // calculate all z's and store in res
        blaze::submatrix(res, 0, 0, NRHS, NX) = blaze::submatrix(rhss, 0, 0, NRHS, NX) * blaze::trans(partialAinv[0]);
        for (std::size_t t = 1; t < NT; ++t) {
            blaze::submatrix(res, 0, t*NX, NRHS, NX) = blaze::submatrix(rhss, 0, t*NX, NRHS, NX) * blaze::trans(partialAinv[t])
                + blaze::submatrix(res, 0, (t-1)*NX, NRHS, NX);
        }
        // now res = z

        // LU-decompose all partial A^{-1} in place
        std::vector<int> ipiv(NX*NT);  // all pivots for all matrices in time-major order
        for (std::size_t t = 0; t < NT-1; ++t) {
            blaze::getrf(partialAinv[t], &ipiv[t*NX]);
        }
        // NT-1 is special
        partialAinv[NT-1] += IdMatrix<std::complex<double>>(NX);
        blaze::getrf(partialAinv[NT-1], &ipiv[(NT-1)*NX]);

        // solve for x
        CDMatrix matLast = blaze::submatrix(res, 0, (NT-1)*NX, NRHS, NX);
        // transpose because LAPACK wants column-major layout
        blaze::getrs(partialAinv[NT-1], matLast, 'T', &ipiv[(NT-1)*NX]);
        blaze::submatrix(res, 0, (NT-1)*NX, NRHS, NX) = matLast;
        for (std::size_t t = 0; t < NT-1; ++t) {
            CDMatrix mat = blaze::submatrix(res, 0, t*NX, NRHS, NX) - matLast;
            blaze::getrs(partialAinv[t], mat, 'T', &ipiv[t*NX]);
            blaze::submatrix(res, 0, t*NX, NRHS, NX) = mat;
        }

#ifndef NDEBUG
        verifyResultOfSolveM(hfm, phi, species, res, rhss);
#endif  // ndef NDEBUG

        return res;
    }

}  // namespace isle
