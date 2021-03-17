# cnspy_trajectory_evaluation

The class `TrajectoryEvaluation` evaluates two trajectories given by a CSV-file (estimated and ground-truth), associates their timestamps, aligns them according to a specified scheme, and computes the absolute trajectory error (ATE) and the normalized estimation error square (NEES). The NEES is based on the uncertainty of the estimated trajectory and the ATE. 
The results can be plotted and will be saved as `EvaluationReport`.

The CSV-file of the estimated trajectory must contain the pose uncertainty (`CSVFormatPose.PoseWithCov`). For the file format please refer to the [spatial_csv_formats]() package and the `CSVFormatPose.py` file. 

## Installation

Install the current code base from GitHub and pip install a link to that cloned copy
```
git clone https://gitlab.aau.at/aau-cns/py3_pkgs/cnspy_trajectory_evaluation.git
cd cnspy_trajectory_evaluation
pip install -e .
```


## Dependencies

* [numpy]()
* [matplotlib]()
* [pandas]()
* [scipy]()
* [timestamp_association]()
* [trajectory]()
* [csv2dataframe]()
* [ros_csv_formats]()

## Definitions and Metrics

As mentioned in the introduction, the aim to evaluate an estimated trajectory of a body reference frame with respect to a global/world reference frame against the true/actual trajectory (the so called groundtruth). Compared to [1], we removed the relative trajectory error (RTE) evaluation, as we agree with the authors that it is less straightforward to compare/judge estimation accuracy. In addition to [1], we added the normalized estimation error square (NEES) evaluation as measure for the estimator's credibility as defined in [2]. The NEES, also known as the Mahalonobis distance squared, is a unit-less metric that relates the absolute estimation error to the estimated uncertainty.  

The estimated quantities can be modeled in various ways, which directly influences the definition of the uncertainty. E.g. assuming we have two coordinate reference frames `G` (GLOBAL) and `B` (BODY). The estimated states are the position and orientation of `B` with respect to `G`. Now regarding the error definition for the position and velocity, one has two options: (i) the position error with respect to the global frame (common case) or (ii) with respect to the body frame. 
As we compute the ATE and the NEES with respect to the global/world reference frame `G`, the uncertainty the position must be expressed in this frame as well. If the uncertainty of the estimator is defined in the body reference frame, it has to be transformed in advance back to the global frame, before the trajectory evaluation is performed. 
In case of the orientation, again various possibilities to define the uncertainty exists. 
First, different representations of orientations exists: rotation matrices in SO(3), unit quaternions in H, or euler angles in radians or degrees. For indirect (error-state) EKF formulations, the use of quaternions has become a gold standard (OpenVINS, LARVIO, VinsMono), while the trend goes towards representing the error in the tangent space of the corresponding manifold (ROVIO).  
Currently, the evaluation tool assumes the orientation uncertainty to refer to the small angle approximations of quaternions `theta`.
Thus, the rotational error for quaternions is defined as `q_err = [1; 0.5 * theta]` or as `R_err = eye(3) + skew(theta)` for SO(3) matrices.
The rotational error is defined as `R_G_B_err = R_G_B_true^T * R_G_B_est`, leading to local perturbations (EQ. 190 in [3]). Note that rotation matrices and unit-quaternions can be mapped directly `R_A_B = R(q_A_B)`, reading as the orientation of `B` with respect to `A`. This means that the uncertainty of the orientation/attitude has to be defined in the local/body reference frame `B`.

## Examples

Please refer to the unit-test section in `TrajectoryEvaluation.py`.

### position error plot

![p_ARMSE](./doc/p_ARMSE.png "folder structure")

### Pose error plot

![pose-err-plot](./doc/pose-err-plot.png "folder structure")

### Pose NEES plot

![pose-nees](./doc/pose-nees.png "folder structure")

## Credits

The classes `AbsoluteTrajectoryError` and `SpatialAlignment` of the  package `trajectory_evaluation` are based on the preliminary work of the  [Robotics and Perception Group, ETH Zurich](http://rpg.ifi.uzh.ch/index.html).


1) From ETH Zurich: [rpg_trajectory_evaluation](https://github.com/uzh-rpg/rpg_trajectory_evaluation) released by Zichao Zhang, Davide Scaramuzza: A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry, IEEE/RSJ Int. Conf. Intell. Robot. Syst. (IROS), 2018.

## References

[1] Z. Zhang and D. Scaramuzza, "A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry," 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Madrid, Spain, 2018, pp. 7244-7251, doi: 10.1109/IROS.2018.8593941.

[2] X. R. Li, Z. Zhao and X. Li, "Evaluation of Estimation Algorithms: Credibility Tests," in IEEE Transactions on Systems, Man, and Cybernetics - Part A: Systems and Humans, vol. 42, no. 1, pp. 147-163, Jan. 2012, doi: 10.1109/TSMCA.2011.2158095.

[3] Joan Solà, "Quaternion kinematics for the error-state Kalman filter", 2017 arXiv, eprint: 1711.02508.

## License

Software License Agreement (GNU GPLv3  License), refer to the LICENSE file.

*Sharing is caring!* - [Roland Jung](https://github.com/jungr-ait)  
