#include <iostream>
#include <random>
#include <blaze/Blaze.h>
#include "../../src/isle/cpp/linear_algebra.hpp"
#include "../../src/isle/cpp/action/hubbardFermiAction.hpp"
#include "../../src/isle/cpp/hubbardFermiMatrixExp.hpp"

#include "../../src/isle/cpp/species.hpp"

#include "catch2/catch.hpp"


TEST_CASE("evalAction",""){
    int Nx = 3;
    int Nt = 3;

    isle::DSparseMatrix kappa (Nx,Nx);

    kappa(0,1) = 1.;
    kappa(0,2) = 1.;
    kappa(1,2) = 1.;
    kappa(1,0) = 1.;
    kappa(2,0) = 1.;
    kappa(2,1) = 1.;

    isle::action::HubbardFermiAction<isle::action::HFAHopping::EXP,isle::action::HFAAlgorithm::DIRECT_SINGLE, isle::action::HFABasis::PARTICLE_HOLE> action_direct_single(kappa,0,-1,false);
    isle::action::HubbardFermiAction<isle::action::HFAHopping::EXP,isle::action::HFAAlgorithm::DIRECT_SQUARE, isle::action::HFABasis::PARTICLE_HOLE> action_direct_square(kappa,0,-1,false);

    std::mt19937 rng{0};
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    isle::DVector phi(Nt*Nx,0);

    for (std::size_t i = 0; i < Nt*Nx; ++i ){
        phi[i] = dist(rng);
    }

    std::cout << "abs(airect_single.force - direct_square.force):\n" << abs(action_direct_single.force(phi) - action_direct_square.force(phi)) << '\n';

    //REQUIRE(abs(action_direct_single.eval(phi) - action_direct_square.eval(phi)) < 1e-15 );
    auto a1 = action_direct_single.force(phi);
    auto a2 = action_direct_square.force(phi);
    for(int i = 0; i < Nx; ++i){
        std::cout << "direct_single.force(" << i << ") = " << a1[i] << ",\ndirect_square.force(" << i << ") = " << a2[i] << std::endl;
        REQUIRE(std::real(a1[i]) == Approx(std::real(a2[i])));
        REQUIRE(std::imag(a1[i]) == Approx(std::imag(a2[i])));
    }
}
