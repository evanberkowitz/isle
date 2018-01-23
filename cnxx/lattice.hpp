/** \file
 * \brief Spacetime lattice.
 */

#ifndef LATTICE_HPP
#define LATTICE_HPP

#include <stdexcept>

#include "core.hpp"
#include "math.hpp"

/// Represents a spacetime lattice.
/**
 * Holds topology and geometry of the spatial lattice. Here; 'site' and 'neighbor'
 * refer to sites on the spatial lattice.
 *
 * Performs consistency checks on all inputs unless `#ndebug == true`.
 * Throws an instance of `std::out_of_range` or `std::invalid_argument` if a check fails.
 */
class Lattice {
public:
    /// Construct with given numbers of sites and neighbors per site.
    Lattice(std::size_t const nt, std::size_t const nx)
        : nTslice{nt}, nSpatial{nx},
          hoppingMat{nx, nx}, distMat{nx} { }

    /// Get the hopping strengths from a given site to all others.
    /**
     * \param site Index of a site (`site < nx()`).
     * \return A view on a row of the hopping strength matrix for the given site.
     */
    auto hopping(const std::size_t site) const noexcept(ndebug) {
#ifndef NDEBUG
        if (!(site < nx()))
            throw std::out_of_range("Site index out of range");
#endif
        return blaze::row(hoppingMat, site);
    }

    /// Get the full hopping matrix.
    SymmetricSparseMatrix<double> &hopping() noexcept {
        return hoppingMat;
    }

    /// Get the full hopping matrix.
    const SymmetricSparseMatrix<double> &hopping() const noexcept {
        return hoppingMat;
    }

    /// Returns true if sites i and j are neighbors.
    bool areNeighbors(const std::size_t i, const std::size_t j) const {
        return hoppingMat.find(i, j) != hoppingMat.end(i);
    }

    /// Set the hopping strength for a pair of sites.
    /**
     * Keeps the hopping matrix symmetric. Connections with zero strength are only
     * inserted if there already is a connection in the matrix.
     *
     * \param i Index of one site.
     * \param j Index of the other site.
     * \param strength Hopping strength between i and j. If set to exactly 0, the
     *                 element is erased from the matrix.
     */
    void setNeighbor(const std::size_t i, const std::size_t j,
                     const double strength) {
#ifndef NDEBUG
        if (!(i < nx()))
            throw std::out_of_range("Index i out of range");
        if (!(j < nx()))
            throw std::out_of_range("Index j out of range");
#endif
        if (strength == 0)
            hoppingMat.erase(i, j);
        else
            hoppingMat.set(i, j, strength);
    }

    /// Get the physical distance between two spatial sites.
    /**
     * \param i `i < nx()`.
     * \param j `j < nx()`.
     * \returns Distance between sites `i` and `j`.
     */
    double distance(const std::size_t i, const std::size_t j) const noexcept(ndebug) {
#ifndef NDEBUG
        if (!(i < nx()))
            throw std::out_of_range("First index out of range");
        if (!(j < nx()))
            throw std::out_of_range("Second index out of range");
#endif
        return distMat(i, j);
    }

    /// Set the physical distance between two spatial sites.
    /**
     * \param i `i < nx()`.
     * \param j `j < nx()`.
     * \param distance Physical distance between sites `i` and `j`.
     */    
    void distance(const std::size_t i, const std::size_t j,
                  const double distance) noexcept(ndebug) {
#ifndef NDEBUG
        if (!(i < nx()))
            throw std::out_of_range("First index out of range");
        if (!(j < nx()))
            throw std::out_of_range("Second index out of range");
#endif
        distMat(i, j) = distance;
    }

    /// Returns the number of time slices.
    std::size_t nt() const noexcept {
        return nTslice;
    }

    /// Returns the number of spatial sites.
    std::size_t nx() const noexcept {
        return nSpatial;
    }

    /// Returns the total lattice size.
    std::size_t lattSize() const noexcept {
        return nSpatial*nTslice;
    }

    /// Returns the matrix of physical distances between spatial sites.
    const SymmetricMatrix<double> &distances() const noexcept {
        return distMat;
    }

private:
    const std::size_t nTslice;  ///< Number of time slices.
    const std::size_t nSpatial;  ///< Number of spatial lattice sites.
    SymmetricSparseMatrix<double> hoppingMat;  ///< Matrix of hopping strengths (`nx() x nx()`).
    SymmetricMatrix<double> distMat;  ///< matrix of physical distances (`nx() x nx()`).
};

#endif  // ndef LATTICE_HPP
