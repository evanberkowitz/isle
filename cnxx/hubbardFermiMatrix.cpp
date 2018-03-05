#include "hubbardFermiMatrix.hpp"

#include <memory>

using namespace std::complex_literals;

namespace {
    /// Resize a square matrix, throws away old elements.
    template <typename MT>
    void resizeMatrix(MT &mat, std::size_t const target) {
#ifndef NDEBUG
        if (mat.rows() != mat.columns())
            throw std::invalid_argument("Matrix is not square.");
#endif
        if (mat.rows() != target)
            mat.resize(target, target, false);
    }
}

/*
 * -------------------------- HubbardFermiMatrix --------------------------
 */

HubbardFermiMatrix::HubbardFermiMatrix(const SymmetricSparseMatrix<double> &kappa,
                                       const Vector<std::complex<double>> &phi,
                                       const double mu,
                                       const std::int8_t sigmaMu,
                                       const std::int8_t sigmaKappa)
    : _kappa{kappa}, _phi{phi}, _mu{mu}, _sigmaMu{sigmaMu}, _sigmaKappa{sigmaKappa}
{
#ifndef NDEBUG
    if (kappa.rows() != kappa.columns())
        throw std::invalid_argument("Hopping matrix is not square.");
#endif
}

HubbardFermiMatrix::HubbardFermiMatrix(const SparseMatrix<double> &kappa,
                                       const Vector<std::complex<double>> &phi,
                                       const double mu,
                                       const std::int8_t sigmaMu,
                                       const std::int8_t sigmaKappa)
    : _kappa{kappa}, _phi{phi}, _mu{mu}, _sigmaMu{sigmaMu}, _sigmaKappa{sigmaKappa}
{
#ifndef NDEBUG
    if (kappa.rows() != kappa.columns())
        throw std::invalid_argument("Hopping matrix is not square.");
#endif
}


void HubbardFermiMatrix::P(SparseMatrix<double> &P) const {
    const std::size_t NX = nx();
    resizeMatrix(P, NX);

    P = _sigmaKappa*_kappa*_kappa - _kappa*(_sigmaKappa*(1 + _mu) + 1 + _sigmaMu*_mu);
    for (std::size_t i = 0; i < NX; ++i)
        P(i, i) += 2 + _sigmaMu*_mu*_mu + (1+_sigmaMu)*_mu;
}

SparseMatrix<double> HubbardFermiMatrix::P() const {
    SparseMatrix<double> p;
    P(p);
    return p;
}


void HubbardFermiMatrix::Q(SparseMatrix<std::complex<double>> &q,
                           const std::size_t tp) const {
    const std::size_t NX = nx();
    const std::size_t NT = nt();
    const std::size_t t = tp==0 ? NT-1 : tp-1;
    const std::int8_t antiPSign = tp==0 ? -1 : 1;  // encode anti-periodic BCs
    resizeMatrix(q, NX);

    for (std::size_t xp = 0; xp < NX; ++xp) {
        for (auto it = _kappa.begin(xp), end = _kappa.end(xp); it != end; ++it) {
            q.set(xp, it->index(), antiPSign*it->value()*_sigmaKappa
                  * std::exp(1.i*_phi[spacetimeCoord(xp, t, NX, NT)]));
        }
        q.set(xp, xp, -antiPSign*(1 + _sigmaMu*_mu)
              * std::exp(1.i*_phi[spacetimeCoord(xp, t, NX, NT)]));
    }
}

SparseMatrix<std::complex<double>> HubbardFermiMatrix::Q(const std::size_t tp) const {
    SparseMatrix<std::complex<double>> q;
    Q(q, tp);
    return q;
}

void HubbardFermiMatrix::Qdag(SparseMatrix<std::complex<double>> &qdag,
                              const std::size_t tp) const {
    const std::size_t NX = nx();
    const std::size_t NT = nt();
    const std::int8_t antiPSign = tp==NT-1 ? -1 : 1;  // encode anti-periodic BCs
    resizeMatrix(qdag, NX);

    for (std::size_t xp = 0; xp < NX; ++xp) {
        for (auto it = _kappa.begin(xp), end = _kappa.end(xp); it != end; ++it) {
            qdag.set(xp, it->index(), antiPSign*it->value()
                     * std::exp(-1.i*_phi[spacetimeCoord(it->index(), tp, NX, NT)]));
        }
        qdag.set(xp, xp, -antiPSign*(1 + _mu)
                 * std::exp(-1.i*_phi[spacetimeCoord(xp, tp, NX, NT)]));
    }
}

SparseMatrix<std::complex<double>> HubbardFermiMatrix::Qdag(const std::size_t tp) const {
    SparseMatrix<std::complex<double>> qdag;
    Qdag(qdag, tp);
    return qdag;
}

void HubbardFermiMatrix::MMdag(SparseMatrix<std::complex<double>> &mmdag) const {
    const std::size_t NX = nx();
    const std::size_t NT = nt();
    resizeMatrix(mmdag, NX*NT);

    auto const p = P();
    SparseMatrix<std::complex<double>> q; // for both Q and Qdag
    for (std::size_t tp = 0; tp < NT; ++tp) {
        blaze::submatrix(mmdag, tp*NX, tp*NX, NX, NX) = p;

        Q(q, tp);
        blaze::submatrix(mmdag, tp*NX, (tp + NT - 1) % NT * NX, NX, NX) += q;

        Qdag(q, tp);
        blaze::submatrix(mmdag, tp*NX, (tp + 1) % NT * NX, NX, NX) += q;
    }
}

SparseMatrix<std::complex<double>> HubbardFermiMatrix::MMdag() const {
    SparseMatrix<std::complex<double>> mmdag;
    MMdag(mmdag);
    return mmdag;
}


void HubbardFermiMatrix::updateKappa(const SymmetricSparseMatrix<double> &kappa) {
    _kappa = kappa;
}

void HubbardFermiMatrix::updateKappa(const SparseMatrix<double> &kappa) {
    _kappa = kappa;
}

void HubbardFermiMatrix::updatePhi(const Vector<std::complex<double>> &phi) {
    _phi = phi;
}


std::size_t HubbardFermiMatrix::nx() const noexcept {
    return _kappa.rows();
}

std::size_t HubbardFermiMatrix::nt() const noexcept {
    return _phi.size() / _kappa.rows();
}

/*
 * -------------------------- LU --------------------------
 */

HubbardFermiMatrix::LU::LU(const std::size_t nt) {
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

bool HubbardFermiMatrix::LU::isConsistent() const {
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

Matrix<std::complex<double>> HubbardFermiMatrix::LU::reconstruct() const {
    const std::size_t nt = dinv.size();
#ifndef NDEBUG
    if (!isConsistent())
        throw std::runtime_error("Components of LU not initialized properly");
#endif

    const std::size_t nx = dinv[0].rows();
    std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(nx);// pivot indices for inversion
    Matrix<std::complex<double>> aux;

    Matrix<std::complex<double>> recon(nx*nt, nx*nt, 0);

    // zeroth row, all columns
    invert(aux=dinv[0], ipiv);
    spacemat(recon, 0, 0, nx) = aux;
    if (nt == 1)
        return recon;
    spacemat(recon, 0, 1, nx) = u[0];
    spacemat(recon, 0, nt-1, nx) = v[0];

    // rows 1 - nt-3 (all 'regular' ones), all columns
    for (std::size_t i = 1; i < nt-2; ++i) {
        invert(aux=dinv[i], ipiv);
        spacemat(recon, i, i, nx) = aux + l[i-1]*u[i-1];
        invert(aux=dinv[i-i], ipiv);
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
    /// Special case LU decomposition for nt == 1.
    HubbardFermiMatrix::LU nt1LU(const HubbardFermiMatrix &hfm) {
        HubbardFermiMatrix::LU lu{1};

        // construct d_0
        SparseMatrix<std::complex<double>> q;
        hfm.Q(q, 0);
        lu.dinv.emplace_back(hfm.P() + q);
        hfm.Qdag(q, 0);
        lu.dinv[0] += q;

        // invert d_0
        std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(hfm.nx());
        invert(lu.dinv[0]);

        return lu;
    }

    /// Special case LU decomposition for nt == 2.
    HubbardFermiMatrix::LU nt2LU(const HubbardFermiMatrix &hfm) {
        const std::size_t nx = hfm.nx();
        constexpr std::size_t nt = 2;
        HubbardFermiMatrix::LU lu{nt};

        const auto P = hfm.P();  // diagonal block P
        SparseMatrix<std::complex<double>> aux0, aux1; // Q, Qdag, and u
        std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(nx);// pivot indices for inversion

        // d_0
        lu.dinv.emplace_back(P);
        invert(lu.dinv[0], ipiv);

        // l_0
        hfm.Q(aux0, 1);
        hfm.Qdag(aux1, 1);
        lu.l.emplace_back((aux0+aux1)*lu.dinv[0]);

        // u_0
        hfm.Q(aux0, 0);
        hfm.Qdag(aux1, 0);
        aux0 += aux1;  // now aux0 = u_0
        lu.u.emplace_back(aux0);
        
        // d_1
        lu.dinv.emplace_back(P - lu.l[0]*aux0);
        invert(lu.dinv[1], ipiv);

        return lu;
    }

    /// General case LU decomposition for nt > 2.
    HubbardFermiMatrix::LU generalLU(const HubbardFermiMatrix &hfm) {
        const std::size_t nx = hfm.nx();
        const std::size_t nt = hfm.nt();
        HubbardFermiMatrix::LU lu{nt};

        const auto P = hfm.P();  // diagonal block P
        SparseMatrix<std::complex<double>> q;  // subdiagonal blocks Q and Q^\dagger

        SparseMatrix<std::complex<double>> u; // previous u, updated along the way
        std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(nx);// pivot indices for inversion

        // starting components of d, u, l
        lu.dinv.emplace_back(P);
        invert(lu.dinv.back(), ipiv);
        hfm.Qdag(u, 0);   // now u = lu.u[0]
        lu.u.emplace_back(u);
        hfm.Q(q, 1);
        lu.l.emplace_back(q*lu.dinv.back());

        // v, h
        hfm.Q(q, 0);
        lu.v.emplace_back(q);
        hfm.Qdag(q, nt-1);
        lu.h.emplace_back(q*lu.dinv.back());

        // iterate for i in [1, nt-3], 'regular' part of d, u, l, v, h
        for (std::size_t i = 1; i < nt-2; ++i) {
            // here, u = lu.u[i-1]
            lu.dinv.emplace_back(P - lu.l[i-1]*u);
            invert(lu.dinv.back(), ipiv);

            hfm.Q(q, i+1);
            lu.l.emplace_back(q*lu.dinv[i]);
            lu.h.emplace_back(-lu.h[i-1]*u*lu.dinv[i]);
            lu.v.emplace_back(-lu.l[i-1]*lu.v[i-1]);

            hfm.Qdag(u, i);
            lu.u.emplace_back(u);  // now u = lu.u[i]
        }
        // from now on u is lu.u[nt-3]

        // additional 'regular' step for d
        lu.dinv.emplace_back(P - lu.l[nt-3]*u);
        invert(lu.dinv.back(), ipiv);

        // final components of u, l
        hfm.Qdag(q, nt-2);
        lu.u.emplace_back(q - lu.l[nt-3]*lu.v[nt-3]);
        hfm.Q(q, nt-1);
        lu.l.emplace_back((q - lu.h[nt-3]*u)*lu.dinv[nt-2]);

        // final component of d
        lu.dinv.emplace_back(P - lu.l[nt-2]*lu.u[nt-2]);
        for (std::size_t i = 0; i < nt-2; ++i)
            lu.dinv[nt-1] -= lu.h[i]*lu.v[i];
        invert(lu.dinv.back(), ipiv);

        return lu;
    }
}

HubbardFermiMatrix::LU getLU(const HubbardFermiMatrix &hfm) {
    switch (hfm.nt()) {
    case 1:
        return nt1LU(hfm);
    case 2:
        return nt2LU(hfm);
    default:
        return generalLU(hfm);
    }
}
        
Vector<std::complex<double>> solve(const HubbardFermiMatrix &hfm,
                                   const Vector<std::complex<double>> &rhs) {
    return solve(getLU(hfm), rhs);
}

Vector<std::complex<double>> solve(const HubbardFermiMatrix::LU &lu,
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

std::complex<double> logdet(const HubbardFermiMatrix &hfm) {
    auto lu = getLU(hfm);
    return ilogdet(lu);
}

std::complex<double> logdet(const HubbardFermiMatrix::LU &lu) {
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

std::complex<double> ilogdet(HubbardFermiMatrix::LU &lu) {
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
