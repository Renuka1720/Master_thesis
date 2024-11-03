import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io, transform
import torch

# TODO: add documentation
class GalaxyZooDataset(Dataset):
    """
    A custom dataset class for loading Galaxy Zoo images and their associated labels.
    """
    def __init__(self, data_directory, extension=".jpeg", label_file=None, transform=None):
        """
        Initializes the GalaxyZooDataset.

        Parameters
        ----------
        data_directory : str
            The directory that contains the images for this dataset.
        extension : str
            The file extension to use when searching for file. '.jpeg'is the default.
        label_file : str
            The name of the file that contains the labels used for training or testing. By default None is specified. In this case no labels will be returned for the individual items!
        transform : callable, optional
            If a transformation is specified it is applied just before returning a sample. None is default.
        """
        self.data_directory = data_directory
        self.transform = transform
        self.files = []
        for file in os.listdir(data_directory):
            if file.endswith(extension):
                self.files.append(os.path.join(data_directory, file))
        self.len = len(self.files)
        self.idstrings = np.zeros(self.len)
        self.labels = torch.Tensor(np.zeros((self.len, 37)))
        if label_file != None:
            data = np.loadtxt(label_file, delimiter=',', skiprows=1)
            ids = data[:,0]
            for i in range(self.len):
                idsstring = self.files[i].split('/')[-1][0:-4]
                row = np.where(ids == int(idsstring))
                self.labels[i] = torch.Tensor(data[row][:,1:])
        for i in range(self.len):
            self.idstrings[i] = int(self.files[i].split('/')[-1][0:-4])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.len

    def __getitem__(self, idx):
        """
        Retrieve a sample (image and the associated label) from the dataset.

        Parameters
        ----------
        idx : int or torch.Tensor
            Index of the sample to fetch. If a tensor is passed, it is converted to a list.

        Returns
        -------
        sample : dict
            A dictionary containing the image, filename, label, and image ID of the sample.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()


        #division by 255 to normalize the values to between 0 and 1
        #swapaxes to get 3x424x424 (PyTorch expects images in the shape (channels, height, width) for most models)
        image = torch.swapaxes(torch.Tensor(io.imread(self.files[idx])/255.0), 0, 2) 
        sample = {'images': image, 'filenames': self.files[idx], 'labels': self.labels[idx], 'id': self.idstrings[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

"""
Example of a sample returned by the GalaxyZooDataset:
{
    'images': torch.Tensor with shape (3, 424, 424),
    'filenames': 'path/to/image_123456.jpeg',
    'labels': tensor([1, 0, 0, ..., 1]),
    'id': 123456
}
"""