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

    // TODO can this work with a SymmetricSparseMatrix?
    /// Set all hopping strengths for a given site.
    /**
     * This function must be called for each site `i` in order of ascending index.
     * It must even be called for sites whose connections have already been specified.
     * Note the condition on the indices under parameter `strengths`!
     *
     * \param i Index of one site.
     * \param strengths Each element is an edge in the connectivity graph represented as
     *                  a pair `(idx, strength)`.<BR>
     *                  `idx` is the 'target' index, i.e. the edge connects sites `i`
     *                  and `idx`. It must satisfy `idx > i`.<BR>
     *                  `strength` is the hopping strength along that connection.<BR>
     *                  Do not pass `strength==0` elements as they clobber up the
     *                  sparse matrix.
     *
     * <B>Example</B>: \code
       Lattice lat{1, 3};
       lat.setNeighbors(0, {{1, 1}});
       lat.setNeighbors(1, {{2, 2}});
     \endcode
     *
     */
//     void setNeighbors(const std::size_t i,
//                       const std::vector<std::pair<std::size_t, double>> &strengths) {
// #ifndef NDEBUG
//         if (!(i < nx()))
//             throw std::out_of_range("Index i out of range");
//         if (strengths.size() >= nx()) // too many includes hopping from i to i
//             throw std::invalid_argument("Too many hopping strengths given");
// #endif
//         hoppingMat.reserve(i, strengths.size());
//         for (const auto &edge : strengths) {
// #ifndef NDEBUG
//             if (std::get<0>(edge) >= nx())
//                 throw std::out_of_range("Target site index out of range");
//             if (std::get<0>(edge) == i)
//                 throw std::invalid_argument("Target index equal to index i");
//             if (std::get<0>(edge) < i)
//                 throw std::invalid_argument("Target index less than index i");
// #endif
//             hoppingMat.append(i, std::get<0>(edge), std::get<1>(edge));
//         }
//         hoppingMat.finalize(i);
//     }

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
