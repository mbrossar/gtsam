/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  LieGroupPreintegration.cpp
 *  @author Luca Carlone
 *  @author Stephen Williams
 *  @author Richard Roberts
 *  @author Vadim Indelman
 *  @author David Jensen
 *  @author Frank Dellaert
 *  @author Martin Brossard
 **/

#include "LieGroupPreintegration.h"

using namespace std;

namespace gtsam {

//------------------------------------------------------------------------------
LieGroupPreintegration::LieGroupPreintegration(
    const boost::shared_ptr<Params>& p, const Bias& biasHat) :
    PreintegrationBase(p, biasHat) {
  resetIntegration();
}

//------------------------------------------------------------------------------
void LieGroupPreintegration::resetIntegration() {
  deltaTij_ = 0.0;
  deltaXij_ = NavState();
  delRdelBiasOmega_.setZero();
  delPdelBiasAcc_.setZero();
  delPdelBiasOmega_.setZero();
  delVdelBiasAcc_.setZero();
  delVdelBiasOmega_.setZero();
}

//------------------------------------------------------------------------------
bool LieGroupPreintegration::equals(const LieGroupPreintegration& other,
    double tol) const {
  return p_->equals(*other.p_, tol) && std::abs(deltaTij_ - other.deltaTij_) < tol
      && biasHat_.equals(other.biasHat_, tol)
      && deltaXij_.equals(other.deltaXij_, tol)
      && equal_with_abs_tol(delRdelBiasOmega_, other.delRdelBiasOmega_, tol)
      && equal_with_abs_tol(delPdelBiasAcc_, other.delPdelBiasAcc_, tol)
      && equal_with_abs_tol(delPdelBiasOmega_, other.delPdelBiasOmega_, tol)
      && equal_with_abs_tol(delVdelBiasAcc_, other.delVdelBiasAcc_, tol)
      && equal_with_abs_tol(delVdelBiasOmega_, other.delVdelBiasOmega_, tol);
}

//------------------------------------------------------------------------------
void LieGroupPreintegration::update(const Vector3& measuredAcc,
    const Vector3& measuredOmega, const double dt, Matrix9* A, Matrix93* B,
    Matrix93* C) {

  // Correct for bias in the sensor frame
  Vector3 acc = biasHat_.correctAccelerometer(measuredAcc);
  Vector3 omega = biasHat_.correctGyroscope(measuredOmega);

  // Possibly correct for sensor pose
  Matrix3 D_correctedAcc_acc, D_correctedAcc_omega, D_correctedOmega_omega;
  if (p().body_P_sensor)
    boost::tie(acc, omega) = correctMeasurementsBySensorPose(acc, omega,
        D_correctedAcc_acc, D_correctedAcc_omega, D_correctedOmega_omega);

  // Save current rotation for updating Jacobians
  const Rot3 oldRij = deltaXij_.attitude();

  // Do update
  deltaTij_ += dt;
  updateFactor(acc, omega, dt, A, B, C); // functional
  if (p().body_P_sensor) {
      //NOT checked
    // More complicated derivatives in case of non-trivial sensor pose
    *C *= D_correctedOmega_omega;
    if (!p().body_P_sensor->translation().isZero())
      *C += *B * D_correctedAcc_omega;
    *B *= D_correctedAcc_acc; // NOTE(frank): needs to be last
  }

  // Update Jacobians
  // TODO(frank): Try same simplification as in new approach
  Matrix3 D_acc_R;
  oldRij.rotate(acc, D_acc_R);
  const Matrix3 D_acc_biasOmega = D_acc_R * delRdelBiasOmega_;

  const Vector3 integratedOmega = omega * dt;
  Matrix3 D_incrR_integratedOmega;
  const Rot3 incrR = Rot3::Expmap(integratedOmega, D_incrR_integratedOmega); // expensive !!
  const Matrix3 incrRt = incrR.transpose();
  delRdelBiasOmega_ = incrRt * delRdelBiasOmega_ - D_incrR_integratedOmega * dt;

  double dt22 = 0.5 * dt * dt;
  const Matrix3 dRij = oldRij.matrix(); // expensive
  delPdelBiasAcc_ += delVdelBiasAcc_ * dt - dt22 * dRij;
  delPdelBiasOmega_ += dt * delVdelBiasOmega_ + dt22 * D_acc_biasOmega;
  delVdelBiasAcc_ += -dRij * dt;
  delVdelBiasOmega_ += D_acc_biasOmega * dt;
}

//------------------------------------------------------------------------------
void LieGroupPreintegration::updateFactor(const Vector3& b_acceleration, const Vector3& b_omega,  const double dt, OptionalJacobian<9, 9> F, OptionalJacobian<9, 3> G1,
    OptionalJacobian<9, 3> G2) {

  Vector9 xi;
  Matrix39 D_xiP_state;
  Vector3 b_v = deltaXij_.bodyVelocity(F ? &D_xiP_state : 0);
  double dt22 = 0.5 * dt * dt;

  // Integrate on tangent space. TODO(frank): coriolis?
  deltaXij_.dR(xi) << dt * b_omega;
  deltaXij_.dP(xi) << dt * b_v + dt22 * b_acceleration;
  deltaXij_.dV(xi) << dt * b_acceleration;

  // Bring back to manifold
  Matrix9 D_newState_xi;
  deltaXij_ = deltaXij_.boxplus(xi, F, G1 || G2 ? &D_newState_xi : 0);

  // Derivative wrt state is computed by retract directly
  // However, as dP(xi) also depends on state, we need to add that contribution
  if (F) {
    F->middleRows<3>(3) += dt * F->block<3,3>(3,3) * D_xiP_state;
  }
  // derivative wrt acceleration
  if (G1) {
    // D_newState_dPxi = D_newState_xi.middleCols<3>(3)
    // D_dPxi_acc = dt22 * I_3x3
    // D_newState_dVxi = D_newState_xi.rightCols<3>()
    // D_dVxi_acc = dt * I_3x3
    // *G2 = D_newState_acc = D_newState_dPxi * D_dPxi_acc + D_newState_dVxi * D_dVxi_acc
    *G1 = D_newState_xi.middleCols<3>(3) * dt22
        + D_newState_xi.rightCols<3>() * dt;
  }
  // derivative wrt omega
  if (G2) {
    // D_newState_dRxi = D_newState_xi.leftCols<3>()
    // D_dRxi_omega = dt * I_3x3
    // *G1 = D_newState_omega = D_newState_dRxi * D_dRxi_omega
    *G2 = D_newState_xi.leftCols<3>() * dt;
  }

}

//------------------------------------------------------------------------------
Vector9 LieGroupPreintegration::biasCorrectedDelta(
    const imuBias::ConstantBias& bias_i, OptionalJacobian<9, 6> H) const {
  // Correct deltaRij, derivative is delRdelBiasOmega_
  const imuBias::ConstantBias biasIncr = bias_i - biasHat_;
  Matrix3 D_correctedRij_bias, D_dR_correctedRij;
  const Vector3 biasInducedOmega = delRdelBiasOmega_ * biasIncr.gyroscope();
  const Rot3 correctedRij = deltaRij().expmap(biasInducedOmega, boost::none,
      H ? &D_correctedRij_bias : 0);
  if (H)
    D_correctedRij_bias *= delRdelBiasOmega_;

  Rot3 R = Rot3::Expmap(biasInducedOmega);
  ExtendedPose3 T0(deltaRij(),deltaPij(),deltaVij());
  Vector9 xi_corr;
  Matrix9 H2, H3;
  xi_corr << delRdelBiasOmega_ * biasIncr.gyroscope(), 
       delPdelBiasAcc_ * biasIncr.accelerometer() + delPdelBiasOmega_ * biasIncr.gyroscope(),
       delVdelBiasAcc_ * biasIncr.accelerometer()
      + delVdelBiasOmega_ * biasIncr.gyroscope();

  ExtendedPose3 Tplus = ExtendedPose3::Expmap(xi_corr, H2);
  Vector9 xi;
  ExtendedPose3 T(correctedRij,
     T0.position() + Tplus.position(),
     T0.velocity() + Tplus.velocity());
  xi = T.boxminus(T);
  Rot3::Logmap(correctedRij, H ? &D_dR_correctedRij : 0);

  if (H) {
    Matrix36 D_dR_bias, D_dP_bias, D_dV_bias;
    Matrix3 Qv = H2.block<3,3>(6,0);
    Matrix3 Qt = H2.block<3,3>(3,0);
    Matrix3 Jw = H2.block<3,3>(0,0);

    D_dR_bias << Z_3x3, D_dR_correctedRij*D_correctedRij_bias;
    D_dP_bias << Jw*delPdelBiasAcc_, Jw*delPdelBiasOmega_ + Qt*delRdelBiasOmega_;
    D_dV_bias << Jw*delVdelBiasAcc_, Jw*delVdelBiasOmega_ + Qv*delRdelBiasOmega_;
    (*H) << D_dR_bias, R.matrix()*D_dP_bias, R.matrix()*D_dV_bias;
    (*H)  = (*H);
  }
  return xi;
}

//------------------------------------------------------------------------------

}// namespace gtsam
