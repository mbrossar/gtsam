/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file EigenOptimizer.cpp
 *
 * @brief optimizer linear factor graph using eigen solver as backend
 *
 * @date Aug 2019
 * @author Mandy Xie
 */

#include <gtsam/base/timing.h>
#include <gtsam/linear/EigenOptimizer.h>
#include <Eigen/MetisSupport>
#include <vector>

using namespace std;

namespace gtsam {
using SpMat = Eigen::SparseMatrix<double>;
/// obtain sparse matrix for eigen sparse solver
std::pair<SpMat, Eigen::VectorXd> obtainSparseMatrix(
    const GaussianFactorGraph &gfg, const Ordering &ordering) {
  gttic_(EigenOptimizer_obtainSparseMatrix);
  // Get sparse entries of Jacobian [A|b] augmented with RHS b.
  auto entries = gfg.sparseJacobian(ordering);
  // Convert boost tuples to Eigen triplets
  vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(entries.size());
  size_t rows = 0, cols = 0;
  for (const auto &e : entries) {
    size_t temp_rows = e.get<0>(), temp_cols = e.get<1>();
    triplets.emplace_back(temp_rows, temp_cols, e.get<2>());
    rows = std::max(rows, temp_rows);
    cols = std::max(cols, temp_cols);
  }
  // ...and make a sparse matrix with it.
  SpMat Ab(rows + 1, cols + 1);
  Ab.setFromTriplets(triplets.begin(), triplets.end());
  Ab.makeCompressed();
  return make_pair<SpMat, Eigen::VectorXd>(Ab.block(0, 0, rows + 1, cols),
                                           Ab.col(cols));
}

/* ************************************************************************* */
VectorValues optimizeEigenQR(const GaussianFactorGraph &gfg,
                             const Ordering &ordering) {
  gttic_(EigenOptimizer_optimizeEigenQR);
  auto Ab_pair = obtainSparseMatrix(gfg, ordering);
  // Solve A*x = b using sparse QR from Eigen
  gttic_(EigenOptimizer_optimizeEigenQR_create_solver);
  Eigen::SparseQR<SpMat, Eigen::NaturalOrdering<int>> solver(Ab_pair.first);
  gttoc_(EigenOptimizer_optimizeEigenQR_create_solver);
  gttic_(EigenOptimizer_optimizeEigenQR_solve);
  Eigen::VectorXd x = solver.solve(Ab_pair.second);
  gttoc_(EigenOptimizer_optimizeEigenQR_solve);
  return VectorValues(x, gfg.getKeyDimMap());
}

VectorValues optimizeEigenQR(const GaussianFactorGraph &gfg,
                                   const std::string &orderingType) {
  if (orderingType == "COLAMD") {
    return optimizeEigenQR(gfg, Ordering::Colamd(gfg));
  } else if (orderingType == "METIS") {
    return optimizeEigenQR(gfg, Ordering::Metis(gfg));
  } else if (orderingType == "NATURAL") {
    return optimizeEigenQR(gfg, Ordering::Natural(gfg));
  } else {
    std::cout
        << "No applicable ordering method, use COLAMD ordering by default !!"
        << std::endl;
    return optimizeEigenQR(gfg, Ordering::Colamd(gfg));
  }
}

/* *************************************************************************/
VectorValues optimizeEigenCholesky(const GaussianFactorGraph &gfg,
                                   const Ordering &ordering) {
  gttic_(EigenOptimizer_optimizeEigenCholesky);
  auto Ab_pair = obtainSparseMatrix(gfg, ordering);
  auto A = Ab_pair.first;
  gttic_(EigenOptimizer_optimizeEigenCholesky_Atranspose);
  auto At = A.transpose();
  gttoc_(EigenOptimizer_optimizeEigenCholesky_Atranspose);
  gttic_(EigenOptimizer_optimizeEigenCholesky_create_solver);
  // Solve A*x = b using sparse QR from Eigen
  Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Eigen::NaturalOrdering<int>>
      solver(At * A);
  gttoc_(EigenOptimizer_optimizeEigenCholesky_create_solver);
  gttic_(EigenOptimizer_optimizeEigenCholesky_solve);
  Eigen::VectorXd x = solver.solve(At * Ab_pair.second);
  gttoc_(EigenOptimizer_optimizeEigenCholesky_solve);
  return VectorValues(x, gfg.getKeyDimMap());
}

VectorValues optimizeEigenCholesky(const GaussianFactorGraph &gfg,
                                   const std::string &orderingType) {
  if (orderingType == "COLAMD") {
    return optimizeEigenCholesky(gfg, Ordering::Colamd(gfg));
  } else if (orderingType == "METIS") {
    return optimizeEigenCholesky(gfg, Ordering::Metis(gfg));
  } else if (orderingType == "NATURAL") {
    return optimizeEigenCholesky(gfg, Ordering::Natural(gfg));
  } else {
    std::cout
        << "No applicable ordering method, use COLAMD ordering by default !!"
        << std::endl;
    return optimizeEigenCholesky(gfg, Ordering::Colamd(gfg));
  }
}

}  // namespace gtsam