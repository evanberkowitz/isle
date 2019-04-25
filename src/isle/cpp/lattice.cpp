#include "lattice.hpp"

/// Type to color graph with.
using color = int;

constexpr color unspecified = 0;  ///< No color set.
constexpr color latA = 1;  ///< On sub graph A, graph B is marked by ~latB.

namespace isle {
    namespace {
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
    }

    bool isBipartite(const SparseMatrix<double> &hoppingMatrix) {
        std::vector<color> colors(hoppingMatrix.rows(), unspecified);
        colors[0] = latA;
        return colorNeighborsDescend(0, colors, hoppingMatrix);
    }

    bool isBipartite(const Lattice &lat) {
        return isBipartite(lat.hopping());
    }
}
