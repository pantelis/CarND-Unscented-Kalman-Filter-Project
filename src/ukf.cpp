#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.65;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.65;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    // State dimension
    n_x_ = 5;

    // Augmented state dimension
    n_aug_ = 7;

    // Sigma point spreading parameter
    lambda_ = 3 - n_aug_;

    // Number of sigma points
    n_sigma_ = 2 * n_aug_ + 1;

    // Weights of sigma points
    weights_ = VectorXd::Zero(n_sigma_);

    // initial state vector
    x_ = VectorXd::Zero(n_x_); // [px, py, v, yaw, yawd]

    // initial covariance matrix
    P_ = MatrixXd::Identity(n_x_, n_x_);

    // Initialize P_ with the one-shot covariance estimate.
    P_(0,0) = 1.; //redundant but instructive
    P_(1,1) = 1.; //redundant but instructive
    P_(2,2) = 2.;
    P_(3,3) = 20.;
    P_(4,4) = 10.;

    // Matrix of predicted sigma points as columns
    Xsig_pred_ = MatrixXd::Zero(n_x_, n_sigma_);

    // at the end of ukf object construction the filter is uninitialized
    is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    if (!is_initialized_) {
        // #R(for radar) meas_rho meas_phi meas_rho_dot timestamp gt_px gt_py gt_vx gt_vy
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

            if (use_radar_) {
                // Convert radar from polar to cartesian coordinates and initialize state.
                // px = rho * cos(phi), py = rho * sin(phi)
                double px = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
                double py = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);
                double vx = meas_package.raw_measurements_[2] * cos(meas_package.raw_measurements_[1]);
                double vy = meas_package.raw_measurements_[2] * sin(meas_package.raw_measurements_[1]);
                x_ << px, py, sqrt(vx*vx+vy*vy), 0., 0.;
            }
        }// #L(for laser) meas_px meas_py timestamp gt_px gt_py gt_vx gt_vy
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

            if (use_laser_) {
                x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1],
                        0.0, 0.0, 0.0;
                }
        }

        // Instead of setting each of the diagonal values to 1, you can try setting the
        // diagonal values by how much difference you expect between the true state
        // and the initialized x state vector. For example, in the project,
        // we assume the standard deviation of the lidar x and y measurements is 0.15.
        // If we initialized p​x with a lidar measurement, the initial variance or
        // uncertainty in px would probably be less than 1.



        // done initializing, no need to predict or update
        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }
    double delta_t =  (meas_package.timestamp_-time_us_) / 1000000.0; // delta_t is in seconds
    time_us_ = meas_package.timestamp_;

    // Predict sigma points
    Prediction(delta_t);

    //#R(for radar) meas_rho meas_phi meas_rho_dot timestamp gt_px gt_py gt_vx gt_vy
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

        if (use_radar_) {
            UpdateRadar(meas_package);
            cout << "Finished Radar Update Step" << endl;
        }

    }// #L(for laser) meas_px meas_py timestamp gt_px gt_py gt_vx gt_vy
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

        if (use_laser_) {
            UpdateLidar(meas_package);
            cout << "Finished Lidar Update Step" << endl;
        }
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

    /////////////////////////////////////////////////////////////////////////
    /// GENERATE AUGMENTED SIGNAL POINTS                                /////
    /////////////////////////////////////////////////////////////////////////
    //create augmented mean vector
    VectorXd x_aug = VectorXd::Zero(n_aug_);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_sigma_);

    //create augmented mean state
    x_aug.head(n_x_) = x_;

    // Covariance of process error - process error consists of two components:
    // longitudinal acceleration error (std_a) and yaw_rate acceleration error (std_yawdd)
    MatrixXd Q = MatrixXd::Zero(2, 2);
    Q << std_a_* std_a_, 0.0,
            0.0, std_yawdd_ * std_yawdd_;

    //create augmented covariance matrix
    P_aug.topLeftCorner(n_x_, n_x_) = P_;

    P_aug.bottomRightCorner(2, 2) = Q;

    //create square root matrix
    MatrixXd A = P_aug.llt().matrixL();

    // Sigma points formulas taken from Probabilistic Robotics book;
    Xsig_aug.col(0) = x_aug;

    for (int i = 1; i < n_aug_+1; i++){
        Xsig_aug.col(i) = x_aug + sqrt(n_aug_ + lambda_) * A.col(i-1);
    }

    for (int i = n_aug_+1; i < n_sigma_; i++){
        Xsig_aug.col(i) = x_aug - sqrt(n_aug_ + lambda_) * A.col(i-n_aug_-1);
    }

    //print Augmented Sigma Points matrix
    //std::cout << "Augmented Sigma Points matrix Xsig_aug = " << std::endl << Xsig_aug << std::endl;

    /////////////////////////////////////////////////////////////////////////
    /// PREDICT SIGMA POINTS                                            /////
    /////////////////////////////////////////////////////////////////////////
    for (int i = 0; i < n_sigma_; i++){
        double px = Xsig_aug(0,i);
        double py = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double psi = Xsig_aug(3,i);
        double psi_dot = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_psi_ddot = Xsig_aug(6,i);

        VectorXd term0 = VectorXd::Zero(n_x_);
        VectorXd term1 = VectorXd::Zero(n_x_);
        VectorXd term2 = VectorXd::Zero(n_x_);

        term0 << px, py, v, psi, psi_dot;

        if (fabs(psi_dot) > 0.001) {
            term1(0) = (v / psi_dot) * (sin(psi + psi_dot * delta_t) - sin(psi));
            term1(1) = (v / psi_dot) * (-cos(psi + psi_dot * delta_t) + cos(psi));
        }
        else {
            term1(0) = v*cos(psi)*delta_t;
            term1(1) = v*sin(psi)*delta_t;
        }
        term1(2) = 0;
        term1(3) = psi_dot * delta_t;
        term1(4) = 0;

        term2(0) = 0.5 * delta_t * delta_t * cos(psi) * nu_a;
        term2(1) = 0.5 * delta_t * delta_t * sin(psi) * nu_a;
        term2(2) = delta_t * nu_a;
        term2(3) = 0.5 * delta_t * delta_t * nu_psi_ddot;
        term2(4) = delta_t * nu_psi_ddot;

        Xsig_pred_.col(i) = term0 + term1 + term2;
    }

    // Print predicted sigma points matrix
    // std::cout << "Predicted signal points matrix Xsig_pred_ = " << std::endl << Xsig_pred_ << std::endl;

    //////////////////////////////////////////////////////////////////////////////
    /// ESTIMATE THE MEAN STATE x_k+1|k AND ITS PROCESS COVARIANCE P_k+1|k  /////
    /////////////////////////////////////////////////////////////////////////////
    // Estimate the state mean and covariance
    weights_(0) = lambda_ / (lambda_ + n_aug_);

    //set weights_
    for (int i = 1; i < n_sigma_; i++){
        weights_(i) = 1.0/(2.0*(lambda_ + n_aug_));
    }

    // predict state mean
    x_ = Xsig_pred_ * weights_;

    //predict state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < n_sigma_; i++){
        VectorXd dx = Xsig_pred_.col(i) - x_;
        wrapAngletoPi(dx(3)); // the index=3 element of the state vector is the angle psi (yaw)
        P_ += weights_(i) * dx * dx.transpose();
    }

    //print result
//    std::cout << "Predicted state" << std::endl;
//    std::cout << x_ << std::endl;
//    std::cout << "Predicted covariance matrix" << std::endl;
//    std::cout << P_ << std::endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

    //set measurement dimension, lidar can measure px and py
    int n_z = 2;

    // input measurements
    VectorXd z = VectorXd::Zero(n_z);
    z = meas_package.raw_measurements_;

    MatrixXd H_laser = MatrixXd::Zero(n_z, n_x_);

    H_laser << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0;

    //mean predicted measurement
    VectorXd z_pred = VectorXd::Zero(n_z);
    VectorXd z_pred_ = H_laser * x_;

    // Innovation
    VectorXd y = z - z_pred_;

    //calculate measurement covariance matrix S
    MatrixXd R = MatrixXd::Zero(n_z, n_z);
    R << std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;

    const MatrixXd PHt = P_ * H_laser.transpose();
    const MatrixXd S = H_laser * PHt + R;
    const MatrixXd K = PHt * S.inverse();

    x_ += K * y;
    P_ -= K * H_laser * P_;

    // NIS for Lidar
    NIS_laser_ = y.transpose() * S.inverse() * y;
    cout << endl << "NIS Laser: " << NIS_laser_ << endl;

}


/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

    //set measurement dimension, radar can measure rho, phi, and rho_dot
    int n_z = 3;

    // input measurements
    VectorXd z = VectorXd::Zero(n_z);
    z = meas_package.raw_measurements_;


    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd::Zero(n_z, n_sigma_);

    //mean predicted measurement
    VectorXd z_pred = VectorXd::Zero(n_z);

    //transform sigma points into measurement space
    for (int i=0; i< n_sigma_; i++){
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double psi = Xsig_pred_(3, i);

        Zsig(0,i) = sqrt(px*px+py*py);
        Zsig(1,i) = atan2(py, px);
        Zsig(2,i) = (px*v*cos(psi) + py*v*sin(psi))/Zsig(0,i);
    }

    //calculate mean predicted measurement
    z_pred = Zsig * weights_;

    // Innovation
    VectorXd y = VectorXd::Zero(n_z);
    y = z - z_pred;
    wrapAngletoPi(y(1)); // phi is the index=1 element of y

    MatrixXd Ysig = MatrixXd::Zero(n_z, n_sigma_);
    for (int i=0; i < n_sigma_; i++){
        Ysig.col(i) = Zsig.col(i) - z_pred;
    }

    // the index=1 row of Ysig is the phi (yaw) angles of the sigma points
    for (int i=0; i < n_sigma_; i++) {
        wrapAngletoPi(Ysig(1, i));
    }
    //measurement covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z, n_z);

    for (int i = 0; i < n_sigma_; i++){
        S += weights_(i) * Ysig.col(i) * Ysig.col(i).transpose();
    }


    // Sensor error covariance matrix
    MatrixXd R = MatrixXd::Zero(n_z, n_z);
    R << std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0, std_radrd_*std_radrd_;

    S += R;

    //print result
    //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
    //std::cout << "S: " << std::endl << S << std::endl;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

    //calculate cross correlation matrix
    for (int i = 0; i < n_sigma_; i++){
        VectorXd dx = Xsig_pred_.col(i) - x_;
        wrapAngletoPi(dx(3)); // the index=3 element of the state vector is the angle psi (yaw)
        Tc += weights_(i) * dx * Ysig.col(i).transpose();
    }

    //calculate Kalman gain K;
    MatrixXd K = MatrixXd::Zero(n_x_, n_z);
    K = Tc * S.inverse();

    // cout << endl << "Kalman Gain: " << K << endl;

    //update state mean and covariance matrix
    x_ += K * y; // y(1) - phi has been wrapped already

    P_ = P_ - K * S * K.transpose();

    // NIS Radar
    NIS_radar_ = y.transpose() * S.inverse() * y;

    //print result
    //std::cout << "Updated state x: " << std::endl << x_ << std::endl;
    //std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;
    cout << endl << "NIS Radar: " << NIS_radar_ << endl;

}
