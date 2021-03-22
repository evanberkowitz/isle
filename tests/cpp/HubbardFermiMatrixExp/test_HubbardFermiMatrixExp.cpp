#include "../../../src/isle/cpp/math.hpp"
#include "../../../src/isle/cpp/species.hpp"
#include "../../../src/isle/cpp/hubbardFermiMatrixExp.hpp"
#include "catch2/catch.hpp"
#include <iostream>

TEST_CASE("HubbardFermiMatrixExp::F", "HFM::F"){
    const std::size_t Nt = 2;
    const std::size_t Nx = 2;
    // we work with kappaTilde = 1
    isle::DMatrix kappaTilde(Nx,Nx);

    for (std::size_t x = 0; x < Nx; ++x) {
        for(std::size_t y = 0; y < Nx; ++y){
            kappaTilde(x,y) = 0;
        }
    }

    // construct matrix with sigmaKappa = -1
    isle::HubbardFermiMatrixExp hfm(/*kappaTilde=*/kappaTilde,/*muTilde=*/0., /*sigmaKappa=*/-1);

    isle::CDVector phi(Nx*Nt);

    for(std::size_t t = 0 ; t < Nt; ++t ){
        for(std::size_t x = 0; x < Nx; ++x){
            phi[t*Nx + x] = std::complex<double>{0,0};
        }
    }

    for (std::size_t t = 0; t < Nt; ++t){

        auto f = hfm.F(t,phi,isle::Species::PARTICLE,false);

        for(std::size_t x = 0; x<Nx;++x){
            for(std::size_t y = 0; y<Nx;++y){
                if (x==y){
                    std::cout << t << " " << x << " " << y << " :"
                    << abs(std::complex<double>{1,0}-f(x,y)) << std::endl;
                    REQUIRE(abs(std::complex<double>{1,0}-f(x,y)) < 1e-15);
                } else {
                    std::cout << t << " " << x << " " << y << " :" << abs(f(x,y)) << std::endl;
                    REQUIRE(abs(f(x,y)) < 1e-15);
                }
            } // for y
        } // for x
    } // for t
} // TEST_CASE("HubbardFermiMatrixExp::F", "HFM::F")
