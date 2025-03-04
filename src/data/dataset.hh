#ifndef __DATASET_HH__
#define __DATASET_HH__

#include <Eigen/Dense>


Eigen::MatrixXd loadImageToMatrix(const std::string &filename, bool &is_rgb, int &width, int &height);

#endif // __DATASET_HH__