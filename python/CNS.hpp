#ifndef cns_HPP
#define cns_HPP

#include <vector>
#include <complex>
#include <iostream>
#include <iomanip>


typedef std::vector<double> d_vec;
typedef std::vector<std::vector<double>> d_mat;
 

//========================= Space ==================================
class Space {
public:
  /*_______members________*/
  const d_mat K;  // Connectivity graph
  const d_mat X;  // Coulomb
  const size_t NL;
  /*_______members________*/

  /*_______constructors________*/
  // default constructor
  Space() = default; // delete
  // destructor
  ~Space() = default;
  // List constructors
  Space(const d_mat &Kij, const d_mat &Xi) : K(Kij), X(Xi), NL(K.size()) {};
  // copy constructor
  Space(const Space &S) = default;
  // move constructor
  Space & operator=(const Space &S) = default;
  // move assignment operator
  Space(Space &&S) = default;
  /*_______constructors________*/

  /*_______methods________*/
  void get_eigensystem(d_vec &eig_vals, d_mat &eig_vecs) const {};
  /*_______methods________*/
};
//========================= Space ==================================


//========================= SpaceTime ==================================
class SpaceTime {
public:
  /*_______members________*/
  const Space space;    // Space
  const size_t NT;  // Number of temporal slices
  const size_t NTOT;
  /*_______members________*/

  /*_______constructors________*/
  // default constructor
  SpaceTime() = default; // delete
  // destructor
  ~SpaceTime() = default;
  // List constructors
  SpaceTime(const Space &S, const size_t NT_in) : 
    space(S), NT(NT_in), NTOT(NT_in*S.NL) {};
  // copy constructor
  SpaceTime(const SpaceTime &ST) = default;
  // move constructor
  SpaceTime & operator=(const SpaceTime &ST) = default;
  // move assignment operator
  SpaceTime(SpaceTime &&ST) = default;
  /*_______constructors________*/
};
//========================= SpaceTime ==================================


//========================= SpaceTimeVector ==================================
class SpaceTimeVector {
public:
  /*_______members________*/
  const SpaceTime space_time;    // Space
  d_vec vec;  // Number of temporal slices
  /*_______members________*/

  /*_______constructors________*/
  // default constructor
  SpaceTimeVector() = default; // delete
  // destructor
  ~SpaceTimeVector() = default;
  // List constructors
  SpaceTimeVector(const SpaceTime &ST, const d_vec &vec_in) : 
    space_time(ST), vec(vec_in) {};
  // copy constructor
  SpaceTimeVector(const SpaceTimeVector &ST_vec) = default;
  // move constructor
  SpaceTimeVector & operator=(const SpaceTimeVector &ST_vec) = default;
  // move assignment operator
  SpaceTimeVector(SpaceTimeVector &&ST_vec) = default;
  /*_______constructors________*/

  /*_______arithmetic operators________*/
  // add vector
  SpaceTimeVector   operator+ (const SpaceTimeVector &ST_vec) const {
    return *this;
  };
  SpaceTimeVector & operator+=(const SpaceTimeVector &ST_vec)      ;
  // subtract vector
  SpaceTimeVector   operator- (const SpaceTimeVector &ST_vec) const;
  SpaceTimeVector & operator-=(const SpaceTimeVector &ST_vec)      ;
  // vector multiplication with scalar
  SpaceTimeVector   operator* (const std::complex<double> &c) const;
  SpaceTimeVector & operator*=(const std::complex<double> &c)      ;
  SpaceTimeVector   operator* (const SpaceTimeVector &ST_vec) const;
  // vector division by scalar
  SpaceTimeVector   operator/ (const std::complex<double> &c) const;
  SpaceTimeVector & operator/=(const std::complex<double> &c)      ;
  /*_______arithmetic operators________*/
};
//========================= SpaceTimeVector ==================================


//========================= AuxiliaryField ==================================
class AuxiliaryField : public SpaceTimeVector{
public:
  AuxiliaryField() = delete;
  // destructor
  ~AuxiliaryField() = default;
  // List constructors
  AuxiliaryField(const SpaceTime &ST, const d_vec &vec_in) : 
    SpaceTimeVector(ST, vec_in){};
  // copy constructor
  AuxiliaryField(const AuxiliaryField &aux) = default;
  // move constructor
  AuxiliaryField & operator=(const AuxiliaryField &aux) = default;
  // move assignment operator
  AuxiliaryField(AuxiliaryField &&aux) = default;
};
//========================= AuxiliaryField ==================================


//========================= FermionMatrix ==================================
class FermionMatrix {
public:
  /*_______members________*/
  const SpaceTime space_time;    // Space
  const double mu;
  const size_t derivative_type;
  /*_______members________*/

  /*_______constructors________*/
  // default constructor
  FermionMatrix() = default; // delete
  // destructor
  ~FermionMatrix() = default;
  // List constructors
  FermionMatrix(const SpaceTime &ST, const double mu_in, const size_t derivative_type_in) : 
    space_time(ST), mu(mu_in), derivative_type(derivative_type_in) {};
  // copy constructor
  FermionMatrix(const FermionMatrix &ST_vec) = default;
  // move constructor
  FermionMatrix & operator=(const FermionMatrix &ST_vec) = default;
  // move assignment operator
  FermionMatrix(FermionMatrix &&ST_vec) = default;
  /*_______constructors________*/

  /*_______methods________*/
  d_mat operator()(const AuxiliaryField &phi){
    d_mat M(space_time.NTOT, d_vec(space_time.NTOT, 1));
    return M;
  };
  const d_mat operator()(const AuxiliaryField &phi) const{
    const d_mat M(space_time.NTOT, d_vec(space_time.NTOT, 1));
    return M;
  };
  double get_det(const AuxiliaryField &phi) const {return 1.0;};
  /*_______methods________*/

};
//========================= FermionMatrix ==================================


//========================= Solver ==================================
class Solver {
public:
  /*_______members________*/
  const FermionMatrix M;
  /*_______members________*/

  /*_______constructors________*/
  // default constructor
  Solver() = default; // delete
  // destructor
  ~Solver() = default;
  // List constructors
  Solver(const FermionMatrix &M_in) : M(M_in) {};
  // copy constructor
  Solver(const Solver &sol) = default;
  // move constructor
  Solver & operator=(const Solver &sol) = default;
  // move assignment operator
  Solver(Solver &&sol) = default;
  /*_______constructors________*/

  /*_______methods________*/
  SpaceTimeVector solve(const SpaceTimeVector &v) const {
    SpaceTimeVector v_out(v);
    return v_out;
  };
  /*_______methods________*/

};
//========================= Solver ==================================


//========================= AuxUpdater ==================================
class AuxUpdater {
public:
  /*_______methods________*/
  virtual void update(AuxiliaryField &phi) const {};
  /*_______methods________*/
};
//========================= AuxUpdater ==================================

//========================= HMCAuxUpdater ==================================
class HMCAuxUpdater : public AuxUpdater {
public:
  /*_______members________*/
  const FermionMatrix M;
  size_t N_HMC{1}, N_MD{1};
  double epsilon{1.0};
  /*_______members________*/

  /*_______constructors________*/
  // default constructor
  HMCAuxUpdater() = default; // delete
  // destructor
  ~HMCAuxUpdater() = default;
  // List constructors
  HMCAuxUpdater(const FermionMatrix &M_in) : M(M_in) {};
  // copy constructor
  HMCAuxUpdater(const HMCAuxUpdater &U) = default;
  // move constructor
  HMCAuxUpdater & operator=(const HMCAuxUpdater &U) = default;
  // move assignment operator
  HMCAuxUpdater(HMCAuxUpdater &&U) = default;
  /*_______constructors________*/

  /*_______methods________*/
  void do_MD_evolution(AuxiliaryField &phi) const {
    M(phi); // do stuff here...
  };
  void update(AuxiliaryField &phi) const {
    const AuxiliaryField phi0(phi);
    do_MD_evolution(phi);
  };
  /*_______methods________*/
};
//========================= HMCAuxUpdater ==================================



#endif /* cns_HPP */
