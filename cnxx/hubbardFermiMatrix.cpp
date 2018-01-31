#include "hubbardFermiMatrix.hpp"
#include <iostream>

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
