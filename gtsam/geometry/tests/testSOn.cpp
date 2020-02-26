/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   testSOn.cpp
 * @brief  Unit tests for dynamic SO(n) classes.
 * @author Frank Dellaert
 **/

#include <gtsam/geometry/SO3.h>
#include <gtsam/geometry/SO4.h>
#include <gtsam/geometry/SOn.h>

#include <gtsam/base/Lie.h>
#include <gtsam/base/Manifold.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/lieProxies.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/testLie.h>
#include <gtsam/nonlinear/Values.h>

#include <CppUnitLite/TestHarness.h>

#include <boost/random.hpp>

#include <iostream>
#include <stdexcept>
#include <type_traits>

using namespace std;
using namespace gtsam;

//******************************************************************************
// Test dhynamic with n=0
TEST(SOn, SO0) {
  const auto R = SOn(0);
  EXPECT_LONGS_EQUAL(0, R.rows());
  EXPECT_LONGS_EQUAL(Eigen::Dynamic, SOn::dimension);
  EXPECT_LONGS_EQUAL(Eigen::Dynamic, SOn::Dim());
  EXPECT_LONGS_EQUAL(0, R.dim());
  EXPECT_LONGS_EQUAL(0, traits<SOn>::GetDimension(R));
}

//******************************************************************************
TEST(SOn, SO5) {
  const auto R = SOn(5);
  EXPECT_LONGS_EQUAL(5, R.rows());
  EXPECT_LONGS_EQUAL(Eigen::Dynamic, SOn::dimension);
  EXPECT_LONGS_EQUAL(Eigen::Dynamic, SOn::Dim());
  EXPECT_LONGS_EQUAL(10, R.dim());
  EXPECT_LONGS_EQUAL(10, traits<SOn>::GetDimension(R));
}

//******************************************************************************
TEST(SOn, Concept) {
  BOOST_CONCEPT_ASSERT((IsGroup<SOn>));
  BOOST_CONCEPT_ASSERT((IsManifold<SOn>));
  BOOST_CONCEPT_ASSERT((IsLieGroup<SOn>));
}

//******************************************************************************
TEST(SOn, CopyConstructor) {
  const auto R = SOn(5);
  const auto B(R);
  EXPECT_LONGS_EQUAL(5, B.rows());
  EXPECT_LONGS_EQUAL(10, B.dim());
}

//******************************************************************************
TEST(SOn, Values) {
  const auto R = SOn(5);
  Values values;
  const Key key(0);
  values.insert(key, R);
  const auto B = values.at<SOn>(key);
  EXPECT_LONGS_EQUAL(5, B.rows());
  EXPECT_LONGS_EQUAL(10, B.dim());
}

//******************************************************************************
TEST(SOn, Random) {
  static boost::mt19937 rng(42);
  EXPECT_LONGS_EQUAL(3, SOn::Random(rng, 3).rows());
  EXPECT_LONGS_EQUAL(4, SOn::Random(rng, 4).rows());
  EXPECT_LONGS_EQUAL(5, SOn::Random(rng, 5).rows());
}

//******************************************************************************
TEST(SOn, HatVee) {
  Vector6 v;
  v << 1, 2, 3, 4, 5, 6;

  Matrix expected2(2, 2);
  expected2 << 0, -1, 1, 0;
  const auto actual2 = SOn::Hat(v.head<1>());
  CHECK(assert_equal(expected2, actual2));
  CHECK(assert_equal((Vector)v.head<1>(), SOn::Vee(actual2)));

  Matrix expected3(3, 3);
  expected3 << 0, -3, 2,  //
      3, 0, -1,           //
      -2, 1, 0;
  const auto actual3 = SOn::Hat(v.head<3>());
  CHECK(assert_equal(expected3, actual3));
  CHECK(assert_equal(skewSymmetric(1, 2, 3), actual3));
  CHECK(assert_equal((Vector)v.head<3>(), SOn::Vee(actual3)));

  Matrix expected4(4, 4);
  expected4 << 0, -6, 5, -3,  //
      6, 0, -4, 2,            //
      -5, 4, 0, -1,           //
      3, -2, 1, 0;
  const auto actual4 = SOn::Hat(v);
  CHECK(assert_equal(expected4, actual4));
  CHECK(assert_equal((Vector)v, SOn::Vee(actual4)));
}

//******************************************************************************
TEST(SOn, RetractLocal) {
  Vector6 v1 = (Vector(6) << 0, 0, 0, 1, 0, 0).finished() / 10000;
  Vector6 v2 = (Vector(6) << 0, 0, 0, 1, 2, 3).finished() / 10000;
  Vector6 v3 = (Vector(6) << 3, 2, 1, 1, 2, 3).finished() / 10000;

  // Check that Cayley yields the same as Expmap for small values
  SOn id(4);
  EXPECT(assert_equal(id.retract(v1), SOn(SO4::Expmap(v1))));
  EXPECT(assert_equal(id.retract(v2), SOn(SO4::Expmap(v2))));
  EXPECT(assert_equal(id.retract(v3), SOn(SO4::Expmap(v3))));

  // Same for SO3:
  SOn I3(3);
  EXPECT(
      assert_equal(I3.retract(v1.tail<3>()), SOn(SO3::Expmap(v1.tail<3>()))));
  EXPECT(
      assert_equal(I3.retract(v2.tail<3>()), SOn(SO3::Expmap(v2.tail<3>()))));

  // If we do expmap in SO(3) subgroup, topleft should be equal to R1.
  Matrix R1 = SO3().retract(v1.tail<3>()).matrix();
  SOn Q1 = SOn::Retract(v1);
  CHECK(assert_equal(R1, Q1.matrix().block(0, 0, 3, 3), 1e-7));
  CHECK(assert_equal(v1, SOn::ChartAtOrigin::Local(Q1), 1e-7));
}

//******************************************************************************
TEST(SOn, vec) {
  Vector10 v;
  v << 0, 0, 0, 0, 1, 2, 3, 4, 5, 6;
  SOn Q = SOn::ChartAtOrigin::Retract(v);
  Matrix actualH;
  const Vector actual = Q.vec(actualH);
  boost::function<Vector(const SOn&)> h = [](const SOn& Q) { return Q.vec(); };
  const Matrix H = numericalDerivative11<Vector, SOn, 10>(h, Q, 1e-5);
  CHECK(assert_equal(H, actualH));
}

//******************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
//******************************************************************************
