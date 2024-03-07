import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io, transform
import torch

# TODO: add documentation
class GalaxyZooDataset(Dataset):
    def __init__(self, data_directory, extension=".jpeg", label_file=None, transform=None):
        """
        Parameters
        ----------
        data_directory : str
            The directory that contains the images for this dataset.
        extension : str
            The file extension to use when searching for file. '.jpeg'is the default.
        label_file : str
            The name of the file that contains the labels used for training of testing. By default None is specified. In this case no labels will be returned for the individual items!
        transform : TODO add correct class name
            I a transformation is specified it is applied just before returning a sample. None is default.
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
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = torch.swapaxes(torch.Tensor(io.imread(self.files[idx])/255.0), 0, 2) #to normalize the RGB values to values between 0 and 1 ,swap 0,2 to get 3x424x424
        sample = {'images': image, 'filenames': self.files[idx], 'labels': self.labels[idx], 'id': self.idstrings[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

# TODO: add a new class for Sebastian's simulation images