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
    const std::size_t t = (tp + NT - 1) % NT;  // (tp-1) and wrap around 0 properly
    resizeMatrix(q, NX);

    for (std::size_t xp = 0; xp < NX; ++xp) {
        for (auto it = _kappa.begin(xp), end = _kappa.end(xp); it != end; ++it) {
            q.set(xp, it->index(), it->value()*_sigmaKappa
                  * std::exp(1.i*_phi[spacetimeCoord(xp, t, NX, NT)]));
        }
        q.set(xp, xp, -(1 + _sigmaMu*_mu)
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
    resizeMatrix(qdag, NX);

    for (std::size_t xp = 0; xp < NX; ++xp) {
        for (auto it = _kappa.begin(xp), end = _kappa.end(xp); it != end; ++it) {
            qdag(xp, it->index()) = it->value()
                * std::exp(-1.i*_phi[spacetimeCoord(it->index(), tp, NX, NT)]);
        }
        qdag.set(xp, xp, - (1+_mu)
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

    const auto P = hfm.P();
    SparseMatrix<std::complex<double>> q;

    std::vector<Matrix<std::complex<double>>> dinv;
    dinv.reserve(nt-2);  // do not need to save the last one (d^-1_{nt-2})
    Matrix<std::complex<double>> aux;
    std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(nx);

    lu.d.emplace_back(P);
    hfm.Qdag(q, 0);
    lu.u.emplace_back(q);
    aux = P;
    blaze::getrf(aux, ipiv.get());
    blaze::getri(aux, ipiv.get());
    dinv.emplace_back(aux);
    hfm.Q(q, 1);
    lu.l.emplace_back(q*aux);

    for (std::size_t i = 1; i < nt-2; ++i) {
        lu.d.emplace_back(P - lu.l[i-1]*lu.u[i-1]);
        hfm.Qdag(q, i);
        lu.u.emplace_back(q);
        hfm.Q(q, i+1);
        aux = lu.d[i];
        blaze::getrf(aux, ipiv.get());
        blaze::getri(aux, ipiv.get());
        dinv.emplace_back(aux);
        lu.l.emplace_back(q*aux);
    }
    lu.d.emplace_back(P - lu.l[nt-3]*lu.u[nt-3]);

    hfm.Q(q, 0);
    lu.v.emplace_back(q);
    hfm.Qdag(q, nt-1);
    lu.h.emplace_back(q*dinv[0]);

    for (std::size_t i = 1; i < nt-2; ++i) {
        lu.v.emplace_back(-lu.l[i-1]*lu.v[i-1]);
        lu.h.emplace_back(-lu.h[i-1]*lu.u[i-1]*dinv[i]);
    }

    hfm.Qdag(q, nt-2);
    lu.u.emplace_back(q - lu.l[nt-3]*lu.v[nt-3]);
    hfm.Q(q, nt-1);
    aux = lu.d[nt-2];
    blaze::getrf(aux, ipiv.get());
    blaze::getri(aux, ipiv.get());
    lu.l.emplace_back((q - lu.h[nt-3]*lu.u[nt-3])*aux);

    lu.d.emplace_back(P - lu.l[nt-2]*lu.u[nt-2]);
    for (std::size_t i = 0; i < nt-2; ++i)
        lu.d[nt-1] -= lu.h[i]*lu.v[i];

    return lu;
}

std::complex<double> logdet(const HubbardFermiMatrix &hfm) {
    return logdet(getLU(hfm));
}

std::complex<double> logdet(const HubbardFermiMatrix::LU &lu) {
    std::complex<double> ldet;

    for (const auto &d : lu.d) {
        ldet += logdet(d);
    }

    return toFirstLogBranch(ldet);
}
