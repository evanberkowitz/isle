#include "lattice.hpp"

#include "bind/logging.hpp"


namespace isle {
    namespace {
        /// Type to color graph with.
        using color = int;

        constexpr color unspecified = 0;  ///< No color set.
        constexpr color latA = 1;  ///< On sub graph A.
        constexpr color latB = ~latA;  ///< On sub graph B.

        /// Color all neighbors of site cur with opposite color of site cur and descent in depth first order.
        bool colorNeighborsDescend(const std::size_t cur, std::vector<color> &colors,
                                   const DSparseMatrix graph) {
            const color col = colors[cur];

            // iterate over neighbors
            for (auto it = graph.cbegin(cur); it != graph.cend(cur); ++it) {
                const auto next = it->index();

                if (colors[next] == unspecified) {
                    colors[it->index()] = ~col;  // use opposite color
                    if (cur < graph.rows()-1  // do not recurse for last row
                        && !colorNeighborsDescend(it->index(), colors, graph))
                        return false;
                }
                else if (colors[next] == col) {
                    // two neighbors have same color => not bipartite
                    return false;
                }
                else {
                    // colors[next] == ~col => already set and fine
                    continue;
                }
            }

            // didn't find conflicts => could be bipartite
            return true;
        }

        bool hasAtlernatingLabels(const std::vector<color> &colors) {
            for (std::size_t i = 0; i < colors.size(); ++i) {
                if (i%2 == 0) {
                    if (colors[i] != latA) {
                        return false;
                    }
                    // else: site color is fine
                }

                // i%2 == 1 implicitly here
                else if (colors[i] != latB) {
                    return false;
                }
                // else color is fine
            }

            return true;
        }
    }

    bool isBipartite(const SparseMatrix<double> &hoppingMatrix) {
        if (hoppingMatrix.rows() == 0) {
            return false;
        }

        std::vector<color> colors(hoppingMatrix.rows(), unspecified);
        // 0 is even => must be colored as latA
        colors[0] = latA;
        const bool bipartite = colorNeighborsDescend(0, colors, hoppingMatrix);

        const bool alternating = hasAtlernatingLabels(colors);
        if (bipartite && !alternating) {
            getLogger("Lattice").info("Lattice is bipartite but site labels do not "
                                      "alternate. All sites with an even index must "
                                      "be on one sublattice and sites with an odd "
                                      "index on the other sublattice. "
                                      "Marking the lattice as 'bot bipartite'.");
        }

        return bipartite && alternating;
    }

    bool isBipartite(const Lattice &lat) {
        return isBipartite(lat.hopping());
    }
}
