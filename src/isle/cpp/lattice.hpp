/** \file
 * \brief Spacetime lattice.
 */

#ifndef LATTICE_HPP
#define LATTICE_HPP

#include <stdexcept>

#include "core.hpp"
#include "math.hpp"

namespace isle {

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
            : _nTslice{nt}, _nSpatial{nx},
              _hoppingMat(nx, nx), _distMat(nx) { }

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
            return blaze::row(_hoppingMat, site);
        }

        /// Get the full hopping matrix.
        SparseMatrix<double> &hopping() noexcept {
            return _hoppingMat;
        }

        /// Get the full hopping matrix.
        const SparseMatrix<double> &hopping() const noexcept {
            return _hoppingMat;
        }

        /// Returns true if sites i and j are neighbors.
        bool areNeighbors(const std::size_t i, const std::size_t j) const {
            return _hoppingMat.find(i, j) != _hoppingMat.end(i);
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
            if (strength == 0) {
                _hoppingMat.erase(i, j);
                _hoppingMat.erase(j, i);
            }
            else {
                _hoppingMat.set(i, j, strength);
                _hoppingMat.set(j, i, strength);
            }
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
            return _distMat(i, j);
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
            _distMat(i, j) = distance;
        }

        /// Returns the number of time slices.
        std::size_t nt() const noexcept {
            return _nTslice;
        }
        /// Returns the number of time slices.
        std::size_t &nt() noexcept {
            return _nTslice;
        }

        /// Returns the number of spatial sites.
        std::size_t nx() const noexcept {
            return _nSpatial;
        }

        /// Returns the total lattice size.
        std::size_t lattSize() const noexcept {
            return _nSpatial*_nTslice;
        }

        /// Returns the matrix of physical distances between spatial sites.
        const SymmetricMatrix<double> &distances() const noexcept {
            return _distMat;
        }

    private:
        std::size_t _nTslice;  ///< Number of time slices.
        const std::size_t _nSpatial;  ///< Number of spatial lattice sites.
        SparseMatrix<double> _hoppingMat;  ///< Matrix of hopping strengths (`nx() x nx()`).
        SymmetricMatrix<double> _distMat;  ///< matrix of physical distances (`nx() x nx()`).
    };

    /// Loop index around boundary.
    /**
     * \param i Index, can be static_cast<std::size_t>(-1) to represent 'one before 0'.
     * \param n `n-1` is the maximum value for `i`.
     *
     * \warning Only works if i is at most one step across boundary.
     *          That is, it requires `-1 <= i <= n`.
     */
    constexpr std::size_t loopIdx(const std::size_t i,
                                  const std::size_t n) noexcept(ndebug) {
        if (i == n)
            return 0;
        if (i == static_cast<std::size_t>(-1))
            return n-1;

#ifndef NDEBUG
        if (i > n)  // i != -1 here, so we can check like this
            throw std::runtime_error("i > n in loopIdx");
#endif

        return i;
    }

    /// Return the flat spacetime coordinate for a given pair of space and time coordinates.
    /**
     * \param x Spatial coordinate.
     * \param t Temporal coordinate.
     * \param nx Number of spatial lattice sites.
     * \param nt Number of temporal lattice sites.
     */
    constexpr std::size_t spacetimeCoord(const std::size_t x,
                                         const std::size_t t,
                                         const std::size_t nx,
                                         const std::size_t UNUSED(nt)) noexcept {
        return t*nx + x;
    }

    /// Return true if hopping matrix of lat is bipartite, false otherwise.
    bool isBipartite(const Lattice &lat);

}  // namespace isle

#endif  // ndef LATTICE_HPP
