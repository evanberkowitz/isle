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
        blaze::submatrix(mmdag, tp*NX, (tp + NT - 1) % NT * NX, NX, NX) = q;

        Qdag(q, tp);
        blaze::submatrix(mmdag, tp*NX, (tp + 1) % NT * NX, NX, NX) = q;
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
    d.reserve(nt);
    u.reserve(nt-1);
    v.reserve(nt-2);
    l.reserve(nt-1);
    h.reserve(nt-2);
}

Matrix<std::complex<double>> HubbardFermiMatrix::LU::reconstruct() const {
    const std::size_t nt = d.size();
#ifndef NDEBUG
    if (nt == 0 || u.size() != nt-1 || v.size() != nt-2
        || l.size() != nt-1 || h.size() != nt-2)
        throw std::runtime_error("Components of LU not fully initialized");
#endif

    const std::size_t nx = d[0].rows();

    Matrix<std::complex<double>> recon(nx*nt, nx*nt, 0);

    // zeroth row, all columns
    blaze::submatrix(recon, 0, 0, nx, nx) = d[0];
    blaze::submatrix(recon, 0, nx, nx, nx) = u[0];
    blaze::submatrix(recon, 0, (nt-1)*nx, nx, nx) = v[0];

    // rows 1 - nt-3 (all 'regular' ones), all columns
    for (std::size_t i = 1; i < nt-2; ++i) {
        blaze::submatrix(recon, i*nx, i*nx, nx, nx) = d[i] + l[i-1]*u[i-1];
        blaze::submatrix(recon, i*nx, (i-1)*nx, nx, nx) = l[i-1]*d[i-1];
        blaze::submatrix(recon, i*nx, (i+1)*nx, nx, nx) = u[i];
        blaze::submatrix(recon, i*nx, (nt-1)*nx, nx, nx) = l[i-1]*v[i-1] + v[i];
    }

    // row nt-2, all columns
    blaze::submatrix(recon, (nt-2)*nx, (nt-2)*nx, nx, nx) = d[nt-2] + l[nt-3]*u[nt-3];
    blaze::submatrix(recon, (nt-2)*nx, (nt-3)*nx, nx, nx) = l[nt-3]*d[nt-3];
    blaze::submatrix(recon, (nt-2)*nx, (nt-1)*nx, nx, nx) = l[nt-3]*v[nt-3] + u[nt-2];

    // row nt-1, up to column nt-3
    blaze::submatrix(recon, (nt-1)*nx, 0, nx, nx) = h[0]*d[0];
    for (std::size_t i = 1; i < nt-2; ++i)
        blaze::submatrix(recon, (nt-1)*nx, i*nx, nx, nx) = h[i-1]*u[i-1] + h[i]*d[i];

    // row nt-1, columns nt-2, nt-1
    blaze::submatrix(recon, (nt-1)*nx, (nt-2)*nx, nx, nx) = h[nt-3]*u[nt-3] + l[nt-2]*d[nt-2];
    blaze::submatrix(recon, (nt-1)*nx, (nt-1)*nx, nx, nx) = d[nt-1] + l[nt-2]*u[nt-2];
    for (std::size_t i = 0; i < nt-2; ++i)
        blaze::submatrix(recon, (nt-1)*nx, (nt-1)*nx, nx, nx) += h[i]*v[i];
    
    return recon;
}

/*
 * -------------------------- free functions --------------------------
 */

HubbardFermiMatrix::LU getLU(const HubbardFermiMatrix &hfm) {
    const std::size_t nx = hfm.nx();
    const std::size_t nt = hfm.nt();
    HubbardFermiMatrix::LU lu{nt};

    const auto P = hfm.P();  // diagonal block P
    SparseMatrix<std::complex<double>> q;  // subdiagonal blocks Q and Q^\dagger

    Matrix<std::complex<double>> dinv;  // inverted d, updated along the way
    SparseMatrix<std::complex<double>> u; // previous u, updated along the way
    std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(nx);// pivot indices for inversion

    // starting components of d, u, l, v, h
    lu.d.emplace_back(P);
    hfm.Qdag(u, 0);   // now u = lu.u[0]
    lu.u.emplace_back(u);

    invert(dinv=P, ipiv);  // now dinv = d[0]^-1

    hfm.Q(q, 1);
    lu.l.emplace_back(q*dinv);
    hfm.Q(q, 0);
    lu.v.emplace_back(q);
    hfm.Qdag(q, nt-1);
    lu.h.emplace_back(q*dinv);

    // iterate for i in [1, nt-3], 'regular' part of d, u, l, v, h
    for (std::size_t i = 1; i < nt-2; ++i) {
        // here, u = lu.u[i-1]
        lu.d.emplace_back(P - lu.l[i-1]*u);

        invert(dinv=lu.d[i], ipiv);  // now dinv = d[i]^-1

        hfm.Q(q, i+1);
        lu.l.emplace_back(q*dinv);
        lu.h.emplace_back(-lu.h[i-1]*u*dinv);
        lu.v.emplace_back(-lu.l[i-1]*lu.v[i-1]);

        hfm.Qdag(u, i);
        lu.u.emplace_back(u);  // now u = lu.u[i]
    }
    // from now on u is lu.u[nt-3]

    // additional 'regular' step for d
    lu.d.emplace_back(P - lu.l[nt-3]*u);

    // final components of u, l
    hfm.Qdag(q, nt-2);
    lu.u.emplace_back(q - lu.l[nt-3]*lu.v[nt-3]);
    hfm.Q(q, nt-1);
    invert(dinv=lu.d[nt-2], ipiv);  // now dinv = d[nt-2]^-1
    lu.l.emplace_back((q - lu.h[nt-3]*u)*dinv);

    // final component of d
    lu.d.emplace_back(P - lu.l[nt-2]*lu.u[nt-2]);
    for (std::size_t i = 0; i < nt-2; ++i)
        lu.d[nt-1] -= lu.h[i]*lu.v[i];

    return lu;
}

template <typename VT>
decltype(auto) spacevec(VT &&vec, const std::size_t t, const std::size_t nx) {
    return blaze::subvector(std::forward<VT>(vec), t*nx, nx);
}

Vector<std::complex<double>> solve(const HubbardFermiMatrix::LU &lu,
                                   const Vector<std::complex<double>> &rhs) {
    const std::size_t nx = lu.d[0].rows();
    const std::size_t nt = lu.d.size();

#ifndef NDEBUG
    if (rhs.size() != nx*nt) {
        throw std::runtime_error("Right hand side does not have correct size (spacetime vector)");
    }
#endif

    Vector<std::complex<double>> y(nt*nx);

    spacevec(y, 0, nx) = spacevec(rhs, 0, nx);
    for (std::size_t i = 1; i < nt-1; ++i)
        spacevec(y, i, nx) = spacevec(rhs, i, nx) - lu.l[i-1]*spacevec(y, i-1, nx);
    spacevec(y, nt-1, nx) = spacevec(rhs, nt-1, nx) - lu.l[nt-2]*spacevec(y, nt-2, nx);
    for (std::size_t j = 0; j < nt-2; ++j)
        spacevec(y, nt-1, nx) -= lu.h[j]*spacevec(y, j, nx);

    Vector<std::complex<double>> x(nt*nx);
    std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(nx);

    Matrix<std::complex<double>> dinv(nx, nx);

    invert(dinv=lu.d[nt-1], ipiv);
    spacevec(x, nt-1, nx) = dinv*spacevec(y, nt-1, nx);

    invert(dinv=lu.d[nt-2], ipiv);
    spacevec(x, nt-2, nx) = dinv*(spacevec(y, nt-2, nx)
                                  - lu.u[nt-2]*spacevec(x, nt-1, nx));

    // iterate i in [nt-3, 0]
    for (std::size_t i = nt-3; i != static_cast<std::size_t>(-1); --i) {
        invert(dinv=lu.d[i], ipiv);
        spacevec(x, i, nx) = dinv*(spacevec(y, i, nx)
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
    std::complex<double> ldet;
    // calculate logdet of diagonal blocks
    for (const auto &d : lu.d)
        ldet += logdet(d);
    return toFirstLogBranch(ldet);
}

std::complex<double> ilogdet(HubbardFermiMatrix::LU &lu) {
    std::complex<double> ldet;
    // calculate logdet of diagonal blocks
    for (auto &d : lu.d)
        ldet += ilogdet(d);
    return toFirstLogBranch(ldet);
}
