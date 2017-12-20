#include "CNS.hpp"

int main(int argc, char** argv){

  // Allocate connectivity graph and other stuff
  const size_t NL{2};
  const d_mat Kij(NL, d_vec(NL, 1));
  const d_mat Xi(3, d_vec(NL, 1));

  // Allocate Space
  const Space S(Kij, Xi);
  std::cout<<"S.NL = "<<S.NL<<std::endl;

  // Space eigensystem infos
  d_vec eig_vals(NL, 0);
  d_mat eig_vecs(NL, eig_vals);
  S.get_eigensystem(eig_vals, eig_vecs);


  // SpaceTime
  const size_t NT(5);
  const SpaceTime ST(S, NT);

  // Member access
  ST.space.get_eigensystem(eig_vals, eig_vecs);

  // SpaceTimeVector
  d_vec vec_in(NT*NL, 1);
  SpaceTimeVector ST_vec(ST, vec_in);
  std::cout<<"ST.NTOT = "<<ST.NTOT<<std::endl;

  // add vectors
  ST_vec + ST_vec;

  // AuxiliaryField
  AuxiliaryField phi(ST, vec_in);
  phi + phi;

  // FermionMatrix
  const double mu(1.0);
  const size_t ND(1);
  FermionMatrix M(ST, mu, ND);

  // Get value for field
  M(phi);
  M.get_det(phi);

  // Solver
  const Solver Sol(M);
  Sol.solve(ST_vec);

  // HMCAuxUpdater
  HMCAuxUpdater U(M);
  size_t N_MD{4}, N_HMC{6};
  double epsilon{0.5};
  U.N_MD    = N_MD;
  U.N_HMC   = N_HMC;
  U.epsilon = epsilon;
  U.update(phi);

  return 0;
}
