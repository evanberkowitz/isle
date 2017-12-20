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
 * Holds topology and geometry of the spatial lattice. Here; 'site' means site on the
 * spatial lattice and 'neighbor' means nearest neighbor on the spatial lattice.
 *
 * A fixed number of neighbors is used for each site. The lattice dimensions cannot be
 * changed after creation.
 *
 * Performs bounds checks on all indices unless `#ndebug == true`.
 * Throws an instance of `std::out_of_range` if a check fails.
 */
class Lattice {
public:
    
    /// Construct with given numbers of sites and neighbors per site.
    Lattice(std::size_t const nt, std::size_t const nx, std::size_t const nNeighbors_)
        : nTslice{nt}, lambda{nt*nx}, neighbors{nx, nNeighbors_},
          strengths{nx, nNeighbors_}, distances{nx} { }

    /// Get the index of a neighbor of a site.
    /**
     * \param site `site < nSpatial()`.
     * \param neigh `neigh < nNeighbors()`.
     * \returns The index of the site of neighbor `neigh` of site `site`.
     */
    std::size_t getNeighbor(std::size_t const site,
                            std::size_t const neigh) const noexcept(ndebug) {
#ifndef NDEBUG
        if (!(neigh < nNeighbors()))
            throw std::out_of_range("Neighbor index out of range");
        if (!(site < nSpatial()))
            throw std::out_of_range("Site index out of range");
#endif
        return neighbors(site, neigh);
    }

    /// Set the index of a neighbor of a site.
    /**
     * \param site `site < nSpatial()`.
     * \param neigh `neigh < nNeighbors()`.
     * \param idx Site index of the neighbor, `idx < nSpatial()`.
     */
    void setNeighbor(std::size_t const site, std::size_t const neigh,
                     std::size_t const idx) noexcept(ndebug) {
#ifndef NDEBUG
        if (!(neigh < nNeighbors()))
            throw std::out_of_range("Neighbor index out of range");
        if (!(site < nSpatial()))
            throw std::out_of_range("Site index out of range");
        if (!(idx < nSpatial()))
            throw std::out_of_range("Index 'idx' out of range");
#endif
        neighbors(site, neigh) = idx;
    }

    /// Get the hopping strength of a connection.
    /**
     * \param site `site < nSpatial()`.
     * \param neigh `neigh < nNeighbors()`.
     * \returns Hopping strength of the link `site <-> Lattice::getNeighbor(site, neigh)`.
     */
    double getStrength(std::size_t const site,
                       std::size_t const neigh) const noexcept(ndebug) {
#ifndef NDEBUG
        if (!(neigh < nNeighbors()))
            throw std::out_of_range("Neighbor index out of range");
        if (!(site < nSpatial()))
            throw std::out_of_range("Site index out of range");
#endif
        return strengths(site, neigh);
    }

    /// Set the hopping strength of a connection.
    /**
     * \param site `site < nSpatial()`.
     * \param neigh `neigh < nNeighbors()`.
     * \param strength Hopping strength of the link `site <-> Lattice::getNeighbor(site, neigh)`.
     */
    void setStrength(std::size_t const site, std::size_t const neigh,
                     double const strength) noexcept(ndebug) {
#ifndef NDEBUG
        if (!(neigh < nNeighbors()))
            throw std::out_of_range("Neighbor index out of range");
        if (!(site < nSpatial()))
            throw std::out_of_range("Site index out of range");
#endif
        strengths(site, neigh) = strength;
    }

    /// Get the physical distance between two spatial sites.
    /**
     * \param i `i < nSpatial()`.
     * \param j `j < nSpatial()`.
     * \returns Distance between sites `i` and `j`.
     */
    double getDistance(std::size_t const i, std::size_t const j) const noexcept(ndebug) {
#ifndef NDEBUG
        if (!(i < nSpatial()))
            throw std::out_of_range("First index out of range");
        if (!(j < nSpatial()))
            throw std::out_of_range("Second index out of range");
#endif
        return distances(i, j);
    }

    /// Set the physical distance between two spatial sites.
    /**
     * \param i `i < nSpatial()`.
     * \param j `j < nSpatial()`.
     * \param distance Physical distance between sites `i` and `j`.
     */    
    void setDistance(std::size_t const i, std::size_t const j,
                     double const distance) noexcept(ndebug) {
#ifndef NDEBUG
        if (!(i < nSpatial()))
            throw std::out_of_range("First index out of range");
        if (!(j < nSpatial()))
            throw std::out_of_range("Second index out of range");
#endif
        distances(i, j) = distance;
    }

    /// Returns the number of spatial sites.
    std::size_t nSpatial() const noexcept {
        return neighbors.rows();
    }

    /// Returns the number of neighbors per site.
    std::size_t nNeighbors() const noexcept {
        return neighbors.columns();
    }

    /// Returns the number of time slices.
    std::size_t nt() const noexcept {
        return nTslice;
    }

    /// Returns the total size of the lattice in space and time.
    std::size_t size() const noexcept {
        return lambda;
    }

    /// Returns the matrix of neighbor indices.
    Matrix<std::size_t> const &neighborMat() const noexcept {
        return neighbors;
    }

    /// Returns the matrix of hopping strengths.
    Matrix<double> const &strengthMat() const noexcept {
        return strengths;
    }

    /// Returns the matrix of physical distances between spatial sites.
    SymmetricMatrix<double> const &distanceMat() const noexcept {
        return distances;
    }

private:
    std::size_t const nTslice;          ///< Number of time slices.
    std::size_t const lambda;           ///< Total size of the lattice.
    Matrix<std::size_t> neighbors;      ///< Matrix of neighbors (`nSpatial() x nNeighbors()`).
    Matrix<double> strengths;           ///< Matrix of hopping strengths (`nSpatial() x nNeighbors()`).
    SymmetricMatrix<double> distances;  ///< matrix of physical distances (`nSpatial() x nSpatial()`).
};

#endif  // ndef LATTICE_HPP
