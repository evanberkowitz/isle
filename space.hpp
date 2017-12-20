/** \file
 * \brief Spatial lattice.
 */

#ifndef SPACE_HPP
#define SPACE_HPP

#include <stdexcept>

#include "core.hpp"
#include "math.hpp"

/// Topology and geometry of a spatial lattice
/**
 * Assigns a fixed number of neighbors to each site. Here, 'Neighbors' always means
 * 'neares neighbors'.
 *
 * Performs bounds checks on all indices unless `#ndebug == true`.
 * Throws an instance of `std::out_of_range` if a check fails.
 */
struct Space {
    Matrix<std::size_t> neighbors;      ///< Matrix of neighbors (`nSites x nNeighbors`).
    Matrix<double> strengths;           ///< Matrix of hopping strengths (`nSites x nNeighbors`).
    SymmetricMatrix<double> distances;  ///< matrix of physical distances (`nSites x nSites`).

    /// Construct with given numbers of sites and neighbors per site.
    Space(std::size_t const nSites_, std::size_t const nNeighbors_)
        : neighbors{nSites_, nNeighbors_},
          strengths{nSites_, nNeighbors_}, distances{nSites_} { }

    /// Get the index of a neighbor of a site.
    /**
     * \param site `site < nSites()`.
     * \param neigh `neigh < nNeighbors()`.
     * \returns The index of the site of neighbor `neigh` of site `site`.
     */
    std::size_t getNeighbor(std::size_t const site,
                            std::size_t const neigh) const noexcept(ndebug) {
#ifndef NDEBUG
        if (!(neigh < nNeighbors()))
            throw std::out_of_range("Neighbor index out of range");
        if (!(site < nSites()))
            throw std::out_of_range("Site index out of range");
#endif
        return neighbors(site, neigh);
    }

    /// Set the index of a neighbor of a site.
    /**
     * \param site `site < nSites()`.
     * \param neigh `neigh < nNeighbors()`.
     * \param idx Site index of the neighbor, `idx < nSites()`.
     */
    void setNeighbor(std::size_t const site, std::size_t const neigh,
                     std::size_t const idx) noexcept(ndebug) {
#ifndef NDEBUG
        if (!(neigh < nNeighbors()))
            throw std::out_of_range("Neighbor index out of range");
        if (!(site < nSites()))
            throw std::out_of_range("Site index out of range");
        if (!(idx < nSites()))
            throw std::out_of_range("Index 'idx' out of range");
#endif
        neighbors(site, neigh) = idx;
    }

    /// Get the hopping strength of a connection.
    /**
     * \param site `site < nSites()`.
     * \param neigh `neigh < nNeighbors()`.
     * \returns Hopping strength of the link `site <-> Space::getNeighbor(site, neigh)`.
     */
    double getStrength(std::size_t const site,
                       std::size_t const neigh) const noexcept(ndebug) {
#ifndef NDEBUG
        if (!(neigh < nNeighbors()))
            throw std::out_of_range("Neighbor index out of range");
        if (!(site < nSites()))
            throw std::out_of_range("Site index out of range");
#endif
        return strengths(site, neigh);
    }

    /// Set the hopping strength of a connection.
    /**
     * \param site `site < nSites()`.
     * \param neigh `neigh < nNeighbors()`.
     * \param strength Hopping strength of the link `site <-> Space::getNeighbor(site, neigh)`.
     */
    void setStrength(std::size_t const site, std::size_t const neigh,
                     double const strength) noexcept(ndebug) {
#ifndef NDEBUG
        if (!(neigh < nNeighbors()))
            throw std::out_of_range("Neighbor index out of range");
        if (!(site < nSites()))
            throw std::out_of_range("Site index out of range");
#endif
        strengths(site, neigh) = strength;
    }

    /// Get the physical distance between two sites.
    /**
     * \param i `i < nSites()`.
     * \param j `j < nSites()`.
     * \returns Distance between sites `i` and `j`.
     */
    double getDistance(std::size_t const i, std::size_t const j) const noexcept(ndebug) {
#ifndef NDEBUG
        if (!(i < nSites()))
            throw std::out_of_range("First index out of range");
        if (!(j < nSites()))
            throw std::out_of_range("Second index out of range");
#endif
        return distances(i, j);
    }

    /// Set the physical distance between two sites.
    /**
     * \param i `i < nSites()`.
     * \param j `j < nSites()`.
     * \param distance Physical distance between sites `i` and `j`.
     */    
    void setDistance(std::size_t const i, std::size_t const j,
                     double const distance) noexcept(ndebug) {
#ifndef NDEBUG
        if (!(i < nSites()))
            throw std::out_of_range("First index out of range");
        if (!(j < nSites()))
            throw std::out_of_range("Second index out of range");
#endif
        distances(i, j) = distance;
    }

    /// Returns the number of sites.
    std::size_t nSites() const noexcept {
        return neighbors.rows();
    }

    /// Returns the number of neighbors per site.
    std::size_t nNeighbors() const noexcept {
        return neighbors.columns();
    }
};

#endif  // ndef SPACE_HPP
