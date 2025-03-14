#ifndef __AUTOENCODER_LINEAR_HH__
#define __AUTOENCODER_LINEAR_HH__

#include <Eigen/Dense>

class Linear{
    public:
        Eigen::MatrixXf weights;
        Eigen::MatrixXf bias;

        Linear(int input_dim, int output_dim);
        Eigen::MatrixXf forward(Eigen::MatrixXf& input);
};

#endif // __AUTOENCODER_LINEAR_HH__