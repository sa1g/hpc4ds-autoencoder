#ifndef __AUTOENCODER_DATASET_HH__
#define __AUTOENCODER_DATASET_HH__

#include <iterator>
#include <Eigen/Dense>
// #include <unsupported/Eigen/CXX11/MatrixXf>

/**
 * @brief Dataloader class - used as iterator you get batches of shape [BATCH_SIZE, HEIGHT * WIDTH]. If
 * there's not enough images, the last batch will be truncated (skipped).
 *
 * @param path Path to the dataset
 * @param filenames Vector of filenames
 * @param width Width of the image
 * @param height Height of the image
 * @param num_images Number of images in the dataset
 * @param batch_size Size of the batch
 * @param shuffle Shuffle the dataset
 */
class Dataloader
{
private:
    std::vector<std::string> _filenames;
    int _width;
    int _height;
    const int _num_images;
    const std::string _path;
    const int _batch_size;
    const std::string _extension{".png"};
    const bool _shuffle;
    int _n_channels{1};
    int _batch_start_index{0};
    int _current_index{0};

    const int num_batches;

    Eigen::MatrixXf _current_batch_data;

public:
    Dataloader(const std::string &path, const std::vector<std::string> filenames, const int width, const int height, const int num_images, const int batch_size, const bool shuffle);

    int get_num_batches() const { return num_batches; }
    bool is_shuffled() const { return _shuffle; }
    int get_num_images() const { return _num_images; }
    std::vector<std::string> get_filenames() const { return _filenames; }

    Eigen::MatrixXf &get_batch();

    class Iterator
    {
    private:
        Dataloader *_dataloader;
        int _current_batch;

    public:
        Iterator(Dataloader *dataloader, int current_batch)
            : _dataloader(dataloader), _current_batch(current_batch) {}

        Eigen::MatrixXf &operator*()
        {
            // Ensure current_batch is within bounds
            if (_dataloader->_batch_start_index >= _dataloader->_num_images)
            {
                throw std::out_of_range("Iterator out of range");
            }

            int batch_start_index = _current_batch * _dataloader->_batch_size;
            if (batch_start_index >= _dataloader->_num_images)
            {
                throw std::out_of_range("Iterator out of range");
            }
            return _dataloader->get_batch();
        }

        Iterator &operator++()
        {
            _dataloader->_batch_start_index += _dataloader->_batch_size;

            // If we reach the end of the dataset, reset the batch index
            if (_dataloader->_batch_start_index >= _dataloader->_num_images)
            {
                _dataloader->_batch_start_index = 0; // Reset to the start of the dataset
            }

            ++_current_batch;
            return *this;
        }

        bool operator!=(const Iterator &other) const
        {
            return _current_batch != other._current_batch;
        }
    };

    // iterator
    Iterator begin() { return Iterator(this, 0); }
    Iterator end() { return Iterator(this, num_batches); }
};

#endif // __AUTOENCODER_DATASET_HH__
