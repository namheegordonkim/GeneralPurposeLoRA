from argparse import ArgumentParser
from os.path import join
from utils import MnistDataloader

import numpy as np
import matplotlib.pyplot as plt


def main():
    training_images_filepath = join(args.mnist_dir, 'train-images-idx3-ubyte')
    training_labels_filepath = join(args.mnist_dir, 'train-labels-idx1-ubyte')
    test_images_filepath = join(args.mnist_dir, 't10k-images-idx3-ubyte')
    test_labels_filepath = join(args.mnist_dir, 't10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Assuming your input tensor is `input_tensor` with shape (B, 28, 28)
    num_patches = 4  # You can change this to the desired number of patches

    n_batches = x_train.shape[0]

    # Calculate the size of each patch
    patch_size = 28 // int(num_patches ** 0.5)
    n_idxs_per_axis = 28 // patch_size

    # Initialize an empty array to store the patches
    reshaped_array = np.empty((n_batches, num_patches, patch_size, patch_size))

    # Extract patches
    patch_idx = 0
    for i in range(n_idxs_per_axis):
        for j in range(n_idxs_per_axis):
            patch = x_train[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            reshaped_array[:, patch_idx] = patch
            patch_idx += 1

    plt.figure()
    plt.imshow(x_train[0], cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(reshaped_array[0, 0], cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(reshaped_array[0, 1], cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(reshaped_array[0, 2], cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(reshaped_array[0, 3], cmap='gray')
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--mnist_dir", type=str, help="The directory containing the MNIST dataset in raw -ubyte format", required=True)
    args = parser.parse_args()
    main()
