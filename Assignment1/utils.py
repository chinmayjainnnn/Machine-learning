import torch
import numpy as np
from typing import Tuple
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import sklearn.model_selection as model_selection
import cv2


def get_data(
        data_path: str = 'data/cifar10_train.npz', is_linear: bool = False,
        is_binary: bool = False, grayscale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load CIFAR-10 dataset from the given path and return the images and labels.
    If is_linear is True, the images are reshaped to 1D array.
    If grayscale is True, the images are converted to grayscale.

    Args:
    - data_path: string, path to the dataset
    - is_linear: bool, whether to reshape the images to 1D array
    - is_binary: bool, whether to convert the labels to binary
    - grayscale: bool, whether to convert the images to grayscale

    Returns:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    '''
    data = np.load(data_path)
    X = data['images']
    try:
        y = data['labels']
    except KeyError:
        y = None
    # print(X.shape)
    X = X.transpose(0, 3, 1, 2)
    # print(X.shape)
    if is_binary:
        idxs0 = np.where(y == 0)[0]
        idxs1 = np.where(y == 1)[0]
        idxs = np.concatenate([idxs0, idxs1])
        X = X[idxs]
        y = y[idxs]
    if grayscale:
        X = convert_to_grayscale(X)
    if is_linear:
        X = X.reshape(X.shape[0], -1)
    if 'cifar' in data_path:
        X_guass = add_guassian_noise_to_images(X, 0.1)
        X_flipped = flip_images(X)
        X = np.concatenate([X, X_guass, X_flipped])
        y = np.concatenate([y, y, y])
    # print(X.shape)
    # print(y.shape)

    # HINT: rescale the images for better (and more stable) learning and performance
    
    return X, y

def add_guassian_noise_to_images(X: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
    '''
    Add gaussian noise to the given images.

    Args:
    - X: np.ndarray, images
    - noise_factor: float, factor of the noise

    Returns:
    - X_noisy: np.ndarray, images with noise
    '''
    # raise NotImplementedError('Add noise to the images here')
    X_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    return X_noisy

def flip_images(X: np.ndarray) -> np.ndarray:
    '''
    Flip the given images horizontally.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels

    Returns:
    - X_flipped: np.ndarray, flipped images
    - y_flipped: np.ndarray, labels
    '''
    # raise NotImplementedError('Flip the images here')
    flipped_batch_images = np.array([cv2.flip(img, 0) for img in X])
    return flipped_batch_images

def convert_to_grayscale(X: np.ndarray) -> np.ndarray:
    '''
    Convert the given images to grayscale.

    Args:
    - X: np.ndarray, images in RGB format

    Returns:
    - X: np.ndarray, grayscale images
    '''
    return np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])


def train_test_split(
        X: np.ndarray, y: np.ndarray, test_ratio: int = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Split the given dataset into training and test sets.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - test_ratio: float, ratio of the test set

    Returns:
    - X_train: np.ndarray, training images
    - y_train: np.ndarray, training labels
    - X_test: np.ndarray, test images
    - y_test: np.ndarray, test labels
    '''
    assert test_ratio < 1 and test_ratio > 0

    # raise NotImplementedError('Split the dataset here')
    print(X.shape[0])
    indexes = np.random.choice(X.shape[0], size=int(X.shape[0]*(1-test_ratio)), replace=False)
    test_indexes = np.array([i for i in range(X.shape[0]) if i not in indexes])
    X_train = X[indexes]
    X_test = X[test_indexes]
    y_train = y[indexes]
    y_test = y[test_indexes]
    return X_train, y_train, X_test, y_test


def get_data_batch(
        X: np.ndarray, y: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Get a batch of the given dataset.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - batch_size: int, size of the batch

    Returns:
    - X_batch: np.ndarray, batch of images
    - y_batch: np.ndarray, batch of labels
    '''
    # idxs = # TODO: get random indices of the batch size without replacement from the dataset
    idxs = np.random.choice(X.shape[0], size=batch_size, replace=False)
    return X[idxs], y[idxs]

# TODO: Read up on generator functions online
def get_contrastive_data_batch1(
        X: np.ndarray, y: np.ndarray, batch_size: int
):  # Yields: Tuple[np.ndarray, np.ndarray, np.ndarray]
    '''
    Get a batch of the given dataset for contrastive learning.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - batch_size: int, size of the batch

    Yields:
    - X_a: np.ndarray, batch of anchor samples
    - X_p: np.ndarray, batch of positive samples
    - X_n: np.ndarray, batch of negative samples
    '''
    start = 0
    indx = np.random.permutation(len(X))
    X = X[indx]
    y = y[indx]
    factor = 16
    batch_size //= factor
    while True:
        # get a batch
        if start + batch_size < X.shape[0]:
            X_batch = X[start:start+batch_size]
            y_batch = y[start:start+batch_size]
        else: 
            X_batch = X[start:]
            y_batch = y[start:]
            start = 0
            indx = np.random.permutation(len(X))
            X = X[indx]
            y = y[indx]
        start += batch_size

        X_a, X_p, X_n = [], [], []
        # print(len(X))
        for inp, out in zip(X_batch, y_batch):
            positive_idxs = np.where(y == out)[0]
            negative_idxs = np.where(y != out)[0]
            # print(positive_idxs)
            positive_sample = np.random.choice(positive_idxs, factor)
            negative_sample = np.random.choice(negative_idxs, factor)
            X_a.extend([inp]*factor)
            X_p.extend(X[positive_sample])
            X_n.extend(X[negative_sample])
        # print(len(X_a), len(X_p), len(X_n))
        yield np.array(X_a), np.array(X_p), np.array(X_n)


# TODO: Read up on generator functions online
def get_contrastive_data_batch(
        X: np.ndarray, y: np.ndarray, batch_size: int
):  # Yields: Tuple[np.ndarray, np.ndarray, np.ndarray]
    '''
    Get a batch of the given dataset for contrastive learning.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - batch_size: int, size of the batch

    Yields:
    - X_a: np.ndarray, batch of anchor samples
    - X_p: np.ndarray, batch of positive samples
    - X_n: np.ndarray, batch of negative samples
    '''
    start = 0
    # indx = np.random.permutation(len(X))
    # X = X[indx]
    # y = y[indx]
    positive_idxs = {}
    negative_idxs = {}
    while True:
        # get a batch
        if start + batch_size < X.shape[0]:
            X_a = X[start:start+batch_size]
            y_a = y[start:start+batch_size]
        else: 
            X_a = X[start:]
            y_a = y[start:]
            start = 0
        start += batch_size

        X_p, X_n = [], []

        for _, out in zip(X_a, y_a):
            if out not in positive_idxs.keys():
                positive_idxs[out] = np.where(y == out)[0]
                print(out, " : ", len(positive_idxs[out]))
            if out not in negative_idxs.keys():
                negative_idxs[out] = np.where(y != out)[0]
                
            positive_sample = np.random.choice(positive_idxs[out], 1)
            negative_sample = np.random.choice(negative_idxs[out], 1)
            X_p.append(X[positive_sample])
            X_n.append(X[negative_sample])
        # print(len(X_a), len(X_p), len(X_n))
        yield np.array(X_a), np.array(X_p), np.array(X_n)

def plot_losses(
        train_losses: list, val_losses: list, title: str
) -> None:
    '''
    Plot the training and validation losses.

    Args:
    - train_losses: list, training losses
    - val_losses: list, validation losses
    - title: str, title of the plot
    '''
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig('images/loss.png')
    plt.close()


def plot_accuracies(
        train_accs: list, val_accs: list, title: str
) -> None:
    '''
    Plot the training and validation accuracies.

    Args:
    - train_accs: list, training accuracies
    - val_accs: list, validation accuracies
    - title: str, title of the plot
    '''
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.savefig('images/acc.png')
    plt.close()


def plot_tsne(
       z: torch.Tensor, y: torch.Tensor 
) -> None:
    '''
    Plot the 2D t-SNE of the given representation.

    Args:
    - z: torch.Tensor, representation
    - y: torch.Tensor, labels
    '''
    z2 = TSNE(n_components=2).fit_transform(z)
    # z2 = # TODO: get 2D t-SNE of the representation
    plt.scatter(z2[:, 0], z2[:, 1], c=y, cmap='tab10')
    plt.savefig('images/tsne.png')
    plt.close()
