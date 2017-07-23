#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

    ///* initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_;

    ///* if this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    ///* if this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    VectorXd x_;

    ///* state covariance matrix
    MatrixXd P_;

    ///* predicted sigma points matrix
    MatrixXd Xsig_pred_;

    ///* time when the state is true, in us
    long long time_us_;

    ///* Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;

    ///* Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;

    ///* Laser measurement noise standard deviation position1 in m
    double std_laspx_;

    ///* Laser measurement noise standard deviation position2 in m
    double std_laspy_;

    ///* Radar measurement noise standard deviation radius in m
    double std_radr_;

    ///* Radar measurement noise standard deviation angle in rad
    double std_radphi_;

    ///* Radar measurement noise standard deviation radius change in m/s
    double std_radrd_;

    ///* Weights of sigma points
    VectorXd weights_;

    ///* State dimension
    int n_x_;

    ///* Augmented state dimension
    int n_aug_;

    ///* Sigma point spreading parameter
    double lambda_;

    ///* Number of Sigma points
    int n_sigma_;

    // NIS Variables for tuning parameters
    double NIS_laser_;
    double NIS_radar_;

    /**
     * Constructor
     */
    UKF();

    /**
     * Destructor
     */
    virtual ~UKF();

    /**
     * ProcessMeasurement
     * @param meas_package The latest measurement data of either radar or laser
     */
    void ProcessMeasurement(MeasurementPackage meas_package);

    /**
     * Prediction Predicts sigma points, the state, and the state covariance
     * matrix
     * @param delta_t Time between k and k+1 in s
     */
    void Prediction(double delta_t);

    /**
     * Updates the state and the state covariance matrix using a laser measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateLidar(MeasurementPackage meas_package);

    /**
     * Updates the state and the state covariance matrix using a radar measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateRadar(MeasurementPackage meas_package);
};

/*  Explanation of the usage of angle wrapping.
    Imagine you're sitting in the car and looking straightforward. Your phi angle is 0.
    If you turn your head to the left 90 degrees, your phi is 90 degrees (or pi/2 rad).
    If you turn your head to the right 90 degrees, you phi is -90 degrees (- pi/2 rad).
    If you keep turning your head to the right and look back, the phi will be -180 degrees (- pi rad).
    If you turn your head to the left until you look back, the phi will be growing until 180 degrees (pi rad).

    So your phi is in the range of [-pi; pi].
   The angle 1.5*pi doesn't make sense. You know that if you keep turning your head to the left,
   the angle will be changing like 0 -> 1/8*pi ->1/2*pi -> 7/8*pi -> pi -> -7/8* pi -> -pi/2 -> -1/8*pi -> 0 -> 1/8*pi.

    From the math in kalman filter, where you sum up angles or substract angles, you might get large angle value.
    Angle normalization brings this value back in the range [-pi; pi].
   */
inline void wrapAngletoPi( double& angle )
{
    double twoPi = 2.0 * M_PI;
    angle = angle - twoPi * floor( (angle+M_PI) / twoPi );
}
#endif /* UKF_H */
