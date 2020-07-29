

/**
 * Example of use of the combinedImuFactor in conjunction with relative pose
 *  - we read IMU and relative pose data from a CSV file, with the following format:
 *  A row starting with "i" is the first initial position formatted with
 *  N, E, D, qW, qX, qY, qZ, velN, velE, velD
 *  A row starting with "0" is an imu measurement
 *  linAccN, linAccE, linAccD, angVelN, angVelE, angVelD
 *  A row starting with "i">0 is a relative pose correction formatted with
 *  N, E, D, qW, qX, qY, qZ
 *
 *  Usage: ./MartinExample2 [data_csv_path] [data_csv_out]
 *  optional arguments:
 *    data_csv_path           path to the CSV file with the IMU data.
 *    data_csv_out            path to the CSV file with the estimated data.
 */

// GTSAM related includes.
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace gtsam;
using namespace std;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class UnaryFactor: public NoiseModelFactor1<Velocity3> {

public:
  /// shorthand for a smart pointer to a factor
  typedef boost::shared_ptr<UnaryFactor> shared_ptr;

  // The constructor requires the variable key, the (X, Y) measurement value,  and the noise model
  UnaryFactor(Key j, const SharedNoiseModel& model):
    NoiseModelFactor1<Velocity3>(model, j) {}

  virtual ~UnaryFactor() {}

  // Using the NoiseModelFactor1 base class there are two functions that must be overridden.
  // The first is the 'evaluateError' function. This function implements the desired measurement
  // function, returning a vector of errors when evaluated at the provided variable value. It
  // must also calculate the Jacobians for this measurement function, if requested.
  Vector evaluateError(const Velocity3& v, boost::optional<Matrix&> H = boost::none) const
  {
    if (H) (*H) = (Matrix(1,3) << 0.0,0.0,1.0).finished();
    return (Vector(1) << v.z()).finished(); 
  }

  // The second is a 'clone' function that allows the factor to be copied. Under most
  // circumstances, the following code that employs the default copy constructor should
  // work fine.
  virtual gtsam::NonlinearFactor::shared_ptr clone() const {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new UnaryFactor(*this))); }

};

string output_filename = "imuFactorExampleResults1.csv";


int main(int argc, char* argv[])
{
  string data_filename;
  if (argc == 1) {
    cerr << "ERROR number of arguments\n";
    return 1;
  } else if (argc == 2){
    data_filename = argv[1];
  } else {
    data_filename = argv[1];
    output_filename = argv[2];
  }

  // Set up output file for plotting errors
  FILE* fp_out = fopen(output_filename.c_str(), "w+");
  fprintf(fp_out, "#time(s),x(m),y(m),z(m),qx,qy,qz,qw,b_gyro_x(rad/s),b_gyro_y(rad/s),b_gyro_z(rad/s),b_acc_x(m/s^2),b_acc_y(m/s^2),b_acc_z(m/s^2)\n");

  // Begin parsing the CSV file. Input the first line for initialization.
  // From there, we'll iterate through the file and we'll preintegrate the IMU
  // or add in the relative pose given the input.
  ifstream file(data_filename.c_str());
  string value;
  double t, dt;
  // Format is (N,E,D,qW,qX,qY,qZ,velN,velE,velD)
  Eigen::Matrix<double,10,1> initial_state = Eigen::Matrix<double,10,1>::Zero();
  getline(file, value, ','); // i

  getline(file, value, ',');
  t = atof(value.c_str());

  for (int i=0; i<9; i++) {
    getline(file, value, ',');
    initial_state(i) = atof(value.c_str());
  }
  getline(file, value, '\n');
  initial_state(9) = atof(value.c_str());
  cout << "initial state:\n" << initial_state.transpose() << "\n\n";

  // Assemble initial quaternion through gtsam constructor :quaternion(w,x,y,z);
  Rot3 prior_rotation = Rot3::Quaternion(initial_state(3), initial_state(4),
                                        initial_state(5), initial_state(6));

  double earth_rate = -1*7.292115e-5;
  Point3 prior_point(initial_state.head<3>());
  Pose3 prior_pose(prior_rotation, prior_point);
  Vector3 prior_velocity(initial_state.tail<3>());

  Vector3 biasAcc(-0.001, 0.001, -0.04);
  Vector3 biasGyro(-5e-5, -3e-5, 7e-5 - 0*5.4e-5);
  imuBias::ConstantBias prior_imu_bias(biasAcc, biasGyro);

  Values initial_values;
  long long int correction_count = 0;
  initial_values.insert(X(correction_count), prior_pose);
  initial_values.insert(V(correction_count), prior_velocity);
  initial_values.insert(B(correction_count), prior_imu_bias);

  // Assemble prior noise model and add it the graph.
  noiseModel::Diagonal::shared_ptr pose_noise_model = noiseModel::Diagonal::Sigmas((Vector(6) << 0.1, 0.1, 0.0001, 0.001, 0.001, 0.001).finished()); // rad,rad,rad,m, m, m
  noiseModel::Diagonal::shared_ptr velocity_noise_model = noiseModel::Isotropic::Sigma(3, 0.1); // m/s
  noiseModel::Diagonal::shared_ptr bias_noise_model = noiseModel::Diagonal::Sigmas((Vector(6) << 1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6).finished());
  noiseModel::Diagonal::shared_ptr velocity_noise_model3 = noiseModel::Isotropic::Sigma(1, 1); // m/s

  // Add all prior factors (pose, velocity, bias) to the graph.
  NonlinearFactorGraph graph;
  graph.add(PriorFactor<Pose3>(X(correction_count), prior_pose, pose_noise_model));
  graph.add(PriorFactor<Vector3>(V(correction_count), prior_velocity,velocity_noise_model));
  graph.add(PriorFactor<imuBias::ConstantBias>(B(correction_count), prior_imu_bias,bias_noise_model));

  // We build the noise model for the IMU factor.
  double gyro_noise_sigma = 2*3.14*2.7/(180*10000);
  double gyro_bias_rw_sigma = 0.2*3.14*2.5/18000000;
  double accel_noise_sigma = 2*0.0016;
  double accel_bias_rw_sigma = 0.2*0.00027;


  Matrix33 measured_acc_cov = Matrix33::Identity(3,3)*pow(accel_noise_sigma,2);
  Matrix33 measured_omega_cov = Matrix33::Identity(3,3)*pow(gyro_noise_sigma,2);
  Matrix33 integration_error_cov = Matrix33::Identity(3,3)*1e-8; // error committed in integrating position from velocities
  Matrix33 bias_acc_cov = Matrix33::Identity(3,3) * pow(accel_bias_rw_sigma,2);
  Matrix33 bias_omega_cov = Matrix33::Identity(3,3) * pow(gyro_bias_rw_sigma,2);
  Matrix66 bias_acc_omega_int = Matrix::Identity(6,6)*1e-8; // error in the bias used for preintegration

  boost::shared_ptr<PreintegratedCombinedMeasurements::Params> p = PreintegratedCombinedMeasurements::Params::MakeSharedD(9.81);
  // PreintegrationBase params:
  p->accelerometerCovariance = measured_acc_cov; // acc white noise in continuous
  p->integrationCovariance = integration_error_cov; // integration uncertainty continuous
  // PreintegratedRotation params:
  p->gyroscopeCovariance = measured_omega_cov; // gyro white noise in continuous
  // PreintegrationCombinedMeasurements params:
  p->biasAccCovariance = bias_acc_cov; // acc bias in continuous
  p->biasOmegaCovariance = bias_omega_cov; // gyro bias in continuous
  p->biasAccOmegaInt = bias_acc_omega_int;
  // should be using 2nd order integration
  p->use2ndOrderCoriolis = true;

  p->omegaCoriolis = (Vector(3) <<-0.659*earth_rate, 0.0,0.752*earth_rate).finished();
 
  std::shared_ptr<PreintegratedCombinedMeasurements> imu_preintegrated_ =
        std::make_shared<PreintegratedCombinedMeasurements>(p, prior_imu_bias);

  assert(imu_preintegrated_); 

  // For simplicity, we use the same noise model for each relative pose factor
  noiseModel::Diagonal::shared_ptr poserelNoise = noiseModel::Diagonal::Sigmas((Vector(6) << 100, 100, 100, 0.25, 0.25, 0.25).finished());
  auto huber = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(1.345), poserelNoise);

  // Create ISAM2
  ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 10;
  parameters.cacheLinearizedFactors = true;
  parameters.findUnusedFactorSlots = true;
  parameters.enableDetailedResults = false;
  parameters.factorization = ISAM2Params::Factorization::QR;

  double lag = 20;
  IncrementalFixedLagSmoother isam(lag, parameters);
  FixedLagSmoother::KeyTimestampMap newTimestamps;

  // Store previous state for the imu integration and the latest predicted outcome.
  NavState prev_state(prior_pose, prior_velocity);
  NavState prop_state = prev_state;
  imuBias::ConstantBias prev_bias = prior_imu_bias;
  double t0 = t;
  t = 0;
  // All priors have been set up, now iterate through the data file.
  while (file.good()) {

    // Parse out first value
    getline(file, value, ',');
    int type = atoi(value.c_str());

    if (type == 0) { // IMU measurement
      Eigen::Matrix<double,6,1> imu = Eigen::Matrix<double,6,1>::Zero();
      getline(file, value, ',');
      dt = atof(value.c_str()) - t - t0;
      t += dt;

      for (int i=0; i<5; ++i) {
        getline(file, value, ',');
        imu(i) = atof(value.c_str());
      }
      getline(file, value, '\n');
      imu(5) = atof(value.c_str());

      // Adding the IMU preintegration.
      imu_preintegrated_->integrateMeasurement(imu.head<3>(), imu.tail<3>(), dt);

    } else if (type >= 1) { // LiDAR relative pose measurement
      Eigen::Matrix<double,7,1> lidar = Eigen::Matrix<double,7,1>::Zero();

      for (int i=0; i<6; ++i) {
        getline(file, value, ',');
        lidar(i) = atof(value.c_str());
      }
      getline(file, value, '\n');
      lidar(6) = atof(value.c_str());

      correction_count++;

      // Adding IMU factor and pose relative factor and optimizing.
      const PreintegratedCombinedMeasurements& preint_imu_combined =
          dynamic_cast<const PreintegratedCombinedMeasurements&>(
            *imu_preintegrated_);
      CombinedImuFactor imu_factor(X(correction_count-1), V(correction_count-1),
          X(correction_count), V(correction_count),
          B(correction_count-1), B(correction_count),
          preint_imu_combined);
      graph.add(imu_factor);

      if(correction_count>1 && correction_count<120){
        graph.add(PriorFactor<Pose3>(X(0), prior_pose, pose_noise_model));
        graph.add(PriorFactor<Vector3>(V(0), prior_velocity,velocity_noise_model));
        bias_noise_model = noiseModel::Diagonal::Sigmas((Vector(6) << 1,1,1,1,1,1).finished());
        graph.add(PriorFactor<imuBias::ConstantBias>(B(0), prior_imu_bias,bias_noise_model));
      }

      // Add lidar factors
      Rot3 deltaRot(lidar(3), lidar(4), lidar(5), lidar(6));
      Point3 deltap(lidar(0), lidar(1), lidar(2));
      Pose3 deltaPose(deltaRot, deltap);

      // Create relative pose factors between consecutive poses
      graph.add(BetweenFactor<Pose3>(X(correction_count-1), X(correction_count), deltaPose, huber));
      graph.add(UnaryFactor(V(correction_count),velocity_noise_model3));

      // Now optimize and compare results.
      prop_state = imu_preintegrated_->predict(prev_state, prev_bias);
      initial_values.insert(X(correction_count), prop_state.pose());
      initial_values.insert(V(correction_count), prop_state.v());
      initial_values.insert(B(correction_count), prev_bias);

      newTimestamps[X(correction_count)] = t;
      newTimestamps[V(correction_count)] = t;
      newTimestamps[B(correction_count)] = t;

      isam.update(graph, initial_values, newTimestamps);
      Values result = isam.calculateEstimate();

      // Overwrite the beginning of the preintegration for the next step.
      prev_state = NavState(result.at<Pose3>(X(correction_count)),
                            result.at<Vector3>(V(correction_count)));
      prev_bias = result.at<imuBias::ConstantBias>(B(correction_count));

      // Reset the preintegration object.
      imu_preintegrated_->resetIntegrationAndSetBias(prev_bias);
      graph.resize(0);
      newTimestamps.clear();
      initial_values.clear();

      

      // print previous estimates
      if(correction_count > 50){
        prev_state = NavState(result.at<Pose3>(X(correction_count-50)),
                              result.at<Vector3>(V(correction_count-50)));
        prev_bias = result.at<imuBias::ConstantBias>(B(correction_count-50));

        // Print out the position and orientation error for comparison.
        Vector3 gtsam_position = prev_state.pose().translation(); 
        Quaternion gtsam_quat = prev_state.pose().rotation().toQuaternion(); 
        Vector3 b_acc = prev_bias.accelerometer();
        Vector3 b_gyro = prev_bias.gyroscope();

        // display position
        cout << correction_count << " Position:" << gtsam_position.transpose() << " " <<  endl;

        fprintf(fp_out, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                t + t0 - 50*0.25, 
                gtsam_position(0), gtsam_position(1), gtsam_position(2),
                gtsam_quat.w(), gtsam_quat.x(), gtsam_quat.y(), gtsam_quat.z(),
                b_gyro(0), b_gyro(1), b_gyro(2),
                b_acc(0), b_acc(1), b_acc(2));
        prev_state = NavState(result.at<Pose3>(X(correction_count)),
                              result.at<Vector3>(V(correction_count)));
        prev_bias = result.at<imuBias::ConstantBias>(B(correction_count));
      }
    } else {
      cerr << "ERROR parsing file\n";
      return 1;
    }
  }
  fclose(fp_out);
  cout << "Complete, results written to " << output_filename << "\n\n";;
  return 0;
}
