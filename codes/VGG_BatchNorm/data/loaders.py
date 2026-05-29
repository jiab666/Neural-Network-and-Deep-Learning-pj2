"""
Data loaders
"""

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CIFAR_ARCHIVE_NAME = "cifar-10-python.tar.gz"
CIFAR_EXTRACTED_DIR = "cifar-10-batches-py"
CIFAR_DOWNLOAD_URLS = [
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://zenodo.org/records/10089977/files/cifar-10-python.tar.gz?download=1",
]
REQUIRED_CIFAR_FILES = [
    "batches.meta",
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
]


def _download_file(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response, open(destination, "wb") as file_obj:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file_obj.write(chunk)


def _has_complete_cifar_dir(directory):
    return all(os.path.exists(os.path.join(directory, filename)) for filename in REQUIRED_CIFAR_FILES)


def _try_copy_existing_dataset(root):
    search_root = Path(root).resolve().parents[4]
    for candidate in search_root.rglob(CIFAR_EXTRACTED_DIR):
        candidate_str = str(candidate)
        if candidate_str == os.path.join(root, CIFAR_EXTRACTED_DIR):
            continue
        if _has_complete_cifar_dir(candidate_str):
            shutil.copytree(candidate_str, os.path.join(root, CIFAR_EXTRACTED_DIR), dirs_exist_ok=True)
            return True
    return False


def ensure_cifar_available(root):
    extracted_dir = os.path.join(root, CIFAR_EXTRACTED_DIR)
    archive_path = os.path.join(root, CIFAR_ARCHIVE_NAME)

    if os.path.isdir(extracted_dir) and _has_complete_cifar_dir(extracted_dir):
        return
    if os.path.isdir(extracted_dir):
        shutil.rmtree(extracted_dir, ignore_errors=True)

    os.makedirs(root, exist_ok=True)
    if _try_copy_existing_dataset(root):
        return

    if os.path.exists(archive_path):
        try:
            with tarfile.open(archive_path, "r:gz") as tar_obj:
                tar_obj.extractall(root)
            if _has_complete_cifar_dir(extracted_dir):
                return
        except (tarfile.TarError, EOFError, OSError):
            try:
                os.remove(archive_path)
            except OSError:
                pass

    last_error = None
    for url in CIFAR_DOWNLOAD_URLS:
        try:
            _download_file(url, archive_path)
            with tarfile.open(archive_path, "r:gz") as tar_obj:
                tar_obj.extractall(root)
            return
        except Exception as exc:
            last_error = exc
            try:
                os.remove(archive_path)
            except OSError:
                pass

    raise RuntimeError(f"Failed to prepare CIFAR-10 dataset in {root}") from last_error


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(root='../data/', batch_size=128, train=True, shuffle=True, num_workers=4, n_items=-1):
    ensure_cifar_available(root)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    data_transforms = transforms.Compose(
        [transforms.ToTensor(),
         normalize])

    dataset = datasets.CIFAR10(root=root, train=train, download=False, transform=data_transforms)
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader


if __name__ == '__main__':
    train_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1, 2, 0])
        plt.imshow(img * 0.5 + 0.5)
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break
