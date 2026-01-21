# Variables
DATA_DIR=data
MNIST_TAR_URL=https://github.com/myleott/mnist_png/raw/refs/heads/master/mnist_png.tar.gz
SVHN_TRAIN_URL=http://ufldl.stanford.edu/housenumbers/train_32x32.mat
SVHN_TEST_URL=http://ufldl.stanford.edu/housenumbers/test_32x32.mat
SVHN_EXTRA_URL=http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
PYTHON_VENV=venv

# Targets

# Default target
all: prepare_mnist prepare_svhn process_svhn cleanup
get_data: prepare_mnist prepare_svhn process_svhn cleanup


# Prepare the MNIST dataset
prepare_mnist:
	@echo "Downloading MNIST dataset..."
	mkdir -p $(DATA_DIR)/mnist
	wget -O $(DATA_DIR)/mnist_png.tar.gz $(MNIST_TAR_URL)
	@echo "Extracting MNIST dataset..."
	tar -xvzf $(DATA_DIR)/mnist_png.tar.gz -C $(DATA_DIR)/mnist
	@echo "Renaming MNIST directories..."
	mv $(DATA_DIR)/mnist/mnist_png/testing $(DATA_DIR)/mnist/test
	mv $(DATA_DIR)/mnist/mnist_png/training $(DATA_DIR)/mnist/train
	rm -rf $(DATA_DIR)/mnist/mnist_png
	rm $(DATA_DIR)/mnist_png.tar.gz

# Prepare the SVHN dataset
prepare_svhn:
	@echo "Downloading SVHN dataset..."
	wget -O $(DATA_DIR)/train_32x32.mat $(SVHN_TRAIN_URL)
	wget -O $(DATA_DIR)/test_32x32.mat $(SVHN_TEST_URL)
	wget -O $(DATA_DIR)/extra_32x32.mat $(SVHN_EXTRA_URL)

# Create Python virtual environment and install requirements
venv: 
	@echo "Creating Python virtual environment without PIP..."
	python3 -m venv --without-pip $(PYTHON_VENV)
	@echo "Downloading PIP-installing script manually..."
	curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py
	@echo "Installing PIP..."
	$(PYTHON_VENV)/bin/python3 get-pip.py
	@echo "Installing dependencies..."
	$(PYTHON_VENV)/bin/pip install -r data/requirements.txt

# Process SVHN images
process_svhn: venv
	@echo "Processing SVHN training set..."
	$(PYTHON_VENV)/bin/python3 data/process_svhn.py --mat-file $(DATA_DIR)/train_32x32.mat --output-dir $(DATA_DIR)/svhn/train
	@echo "Processing SVHN test set..."
	$(PYTHON_VENV)/bin/python3 data/process_svhn.py --mat-file $(DATA_DIR)/test_32x32.mat --output-dir $(DATA_DIR)/svhn/test
	@echo "Processing SVHN extra set..."
	$(PYTHON_VENV)/bin/python3 data/process_svhn.py --mat-file $(DATA_DIR)/extra_32x32.mat --output-dir $(DATA_DIR)/svhn/train

# Cleanup the dataset files and virtual environment
cleanup:
	@echo "Cleaning up downloaded files..."
	rm $(DATA_DIR)/train_32x32.mat
	rm $(DATA_DIR)/test_32x32.mat
	rm $(DATA_DIR)/extra_32x32.mat
	@echo "Removing virtual environment..."
	rm -rf $(PYTHON_VENV)
	rm get-pip.py

# Clean up the dataset files
clean_data:
	@echo "Removing the datasets..."
	rm -rf $(DATA_DIR)/mnist
	rm -rf $(DATA_DIR)/svhn
	@echo "Done!"

