/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    testLieGroupPreintegration.cpp
 * @brief   Unit test for the LieGroupPreintegration
 * @author  Luca Carlone
 */

#include <gtsam/navigation/LieGroupPreintegration.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/nonlinear/expressions.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/expressionTesting.h>

#include <CppUnitLite/TestHarness.h>
#include <boost/bind.hpp>

#include "imuFactorTesting.h"

namespace testing {
// Create default parameters with Z-down and above noise parameters
static boost::shared_ptr<PreintegrationParams> Params() {
  auto p = PreintegrationParams::MakeSharedD(kGravity);
  p->gyroscopeCovariance = kGyroSigma * kGyroSigma * I_3x3;
  p->accelerometerCovariance = kAccelSigma * kAccelSigma * I_3x3;
  p->integrationCovariance = 0.0001 * I_3x3;
  return p;
}
}

/* ************************************************************************* */
TEST(LieGroupPreintegration, BiasCorrectionJacobians) {
  testing::SomeMeasurements measurements;

  boost::function<Rot3(const Vector3&, const Vector3&)> deltaRij =
      [=](const Vector3& a, const Vector3& w) {
        LieGroupPreintegration pim(testing::Params(), Bias(a, w));
        testing::integrateMeasurements(measurements, &pim);
        return pim.deltaRij();
      };

  boost::function<Point3(const Vector3&, const Vector3&)> deltaPij =
      [=](const Vector3& a, const Vector3& w) {
        LieGroupPreintegration pim(testing::Params(), Bias(a, w));
        testing::integrateMeasurements(measurements, &pim);
        return pim.deltaPij();
      };

  boost::function<Vector3(const Vector3&, const Vector3&)> deltaVij =
      [=](const Vector3& a, const Vector3& w) {
        LieGroupPreintegration pim(testing::Params(), Bias(a, w));
        testing::integrateMeasurements(measurements, &pim);
        return pim.deltaVij();
      };

  // Actual pre-integrated values
  LieGroupPreintegration pim(testing::Params());
  testing::integrateMeasurements(measurements, &pim);

  EXPECT(
      assert_equal(numericalDerivative21(deltaRij, kZero, kZero),
          Matrix3(Z_3x3)));
  EXPECT(
      assert_equal(numericalDerivative22(deltaRij, kZero, kZero),
          pim.delRdelBiasOmega(), 1e-3));

  EXPECT(
      assert_equal(numericalDerivative21(deltaPij, kZero, kZero),
          pim.delPdelBiasAcc()));
  EXPECT(
      assert_equal(numericalDerivative22(deltaPij, kZero, kZero),
          pim.delPdelBiasOmega(), 1e-3));

  EXPECT(
      assert_equal(numericalDerivative21(deltaVij, kZero, kZero),
          pim.delVdelBiasAcc()));
  EXPECT(
      assert_equal(numericalDerivative22(deltaVij, kZero, kZero),
          pim.delVdelBiasOmega(), 1e-3));
}

/* ************************************************************************* */
TEST(LieGroupPreintegration, computeError) {
  LieGroupPreintegration pim(testing::Params());
  NavState x1, x2;
  imuBias::ConstantBias bias;
  Matrix9 aH1, aH2;
  Matrix96 aH3;
  pim.computeError(x1, x2, bias, aH1, aH2, aH3);
  boost::function<Vector9(const NavState&, const NavState&,
                          const imuBias::ConstantBias&)> f =
      boost::bind(&LieGroupPreintegration::computeError, pim, _1, _2, _3,
                  boost::none, boost::none, boost::none);
  // NOTE(frank): tolerance of 1e-3 on H1 because approximate away from 0
  EXPECT(assert_equal(numericalDerivative31(f, x1, x2, bias), aH1, 1e-9));
  EXPECT(assert_equal(numericalDerivative32(f, x1, x2, bias), aH2, 1e-9));
  EXPECT(assert_equal(numericalDerivative33(f, x1, x2, bias), aH3, 1e-9));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
