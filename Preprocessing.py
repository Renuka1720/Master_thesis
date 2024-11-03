import numpy
import copy

import torch
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as transforms


class DielemanTransformation():
    """
    Applies image transformations for data augmentation, exploiting galaxy image invariances as described by Dieleman et al. 2015.
    """
    def __init__(self, rotation_range, translation_range, scaling_range, flip):
        """
        Initializes the DielemanTransformation class with the specified ranges for rotation, translation, scaling, and 
        flipping transformations.
        
        Parameters
        ----------
        rotation_range : tuple
            The range (in degrees) for random rotation.
        translation_range : tuple
            The translation range (fraction of total image size) for random shifts.
        scaling_range : tuple
            Range of scaling factors for zooming in/out.
        flip : float
            Probability of flipping the image horizontally.
        """
        self.scaling_range = scaling_range
        self.random_affine = transforms.RandomAffine(degrees=rotation_range, translate=translation_range, shear=None)
        self.flip = transforms.RandomHorizontalFlip(p=flip)

    def __call__(self, x):
        """
        Applies the transformations to the input image.

        1. Random affine transformation: Applies rotation and translation to the input image.
        2. Random zoom: Applies a zoom effect by scaling the image size based on a random factor sampled 
           between the `scaling_range`.
        3. Random horizontal flip: Flips the image horizontally with the probability specified by `flip`.

        Parameters
        ----------
        x : dict
            A dictionary containing the key 'images', where the value is a tensor representing 
            the input image.

        Returns
        -------
        dict
            A dictionary with the transformed image under the 'images' key.
        """
        input_image = x['images']

        # Apply the random affine transformation (rotation + translation)
        transformed_image = self.random_affine.__call__(input_image)

        #Samples a zoom factor using log-uniform distribution for balanced zooming, then applies exponentiation to return it to a linear scale.
        zoom = numpy.exp(numpy.random.uniform(numpy.log(self.scaling_range[0]), numpy.log(self.scaling_range[1])))
        resize = TF.resize(transformed_image, (int(input_image.shape[1]*zoom),int(input_image.shape[2]*zoom)), antialias=True)
        x['images'] = self.flip.__call__(resize)
        return x

class AlexnetTransformation():
    def __init__(self, resize, centercrop, mean, std):
        self.data_transform = transforms.Compose([  #transforms.Resize(resize, antialias=False),
                                                    transforms.CenterCrop(centercrop),
                                                    #transforms.Normalize(mean=mean, std=std)
                                                    ])

    def __call__(self, x):
        x['images'] = self.data_transform.__call__(x['images'])
        return x

# class Vgg16Transformation():
#     def __init__(self, resize, centercrop, mean, std):
#         self.data_transform = transforms.Compose([
#                                                     transforms.Resize((224, 224)),
#                                                     transforms.RandomHorizontalFlip(),
#                                                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
#                                                     transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
#                                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#                                                  ])


    # def __call__(self, x):
    #     x['images'] = self.data_transform.__call__(x['images'])
    #     return x

    
class KrizhevskyColorTransformation():
    
    def __init__(self, weights, std):
        """
        Performs a random brightness scaling based on the specified weights and standard deviation following the PCA based idea by Alex Krizhevsky et al. 2012 as used by Dieleman et al. 2015.

        Parameters
        ----------
        weights : list 
            weights=[-0.0148366, -0.01253134, -0.01040762]
            The weight values represent how much each channel contributes to that particular principal component.
            These weights are used to scale the random noise added to each color channel.
    
        std : float
            Standard deviation (amount of randomness) used when generating noise.
        """
        self.weights = torch.tensor(numpy.array(weights))
        self.std = std

    def __call__(self, x):
        """
        Parameters
        ----------
        x : dictonary
            A dictonary conatining the data items returned from the data set. They key 'image' refers to an array of RGB values in the range between 0 and 1.
        
        Returns
        -------
        dictonary
            modifies the entered dictonary and changes the 'image' array accordingly. Ensures that the returned values of the image are between 0 and 1
        """
        transformed_image = x['images']
        noise = torch.normal(0.0, self.std, size=[1]) * self.weights       #noise being scaled by the weights
        transformed_image[0] = transformed_image[0] + noise[0]             #red pixel
        transformed_image[1] = transformed_image[1] + noise[1]             #green pixel
        transformed_image[2] = transformed_image[2] + noise[2]             #blue pixel
        transformed_image = torch.clip(transformed_image, 0, 1)
        x['images'] = transformed_image
        return x
    
class NoViewpointTransformation():
    def __init__(self, output_size, downsampling_factor):
        self.output_size = output_size
        self.downsampling_factor = downsampling_factor

    def __call__(self, x):
        """
        Applies center cropping and resizing to the input image.

        Args:
            x (dict): Input dictionary containing the image.

        Returns:
            dict: Output dictionary with the transformed image.
        """
        transformed_image = x['images']
        transformed_image = transformed_image.reshape(-1, transformed_image.shape[-3], transformed_image.shape[-2], transformed_image.shape[-1])

        # Center crop and resize
        cropped_image = TF.center_crop(transformed_image, [int(self.downsampling_factor * i) for i in self.output_size])
        transformed_image = TF.resize(cropped_image, self.output_size, antialias=True)

        # Move channels to second dimension and reshape
        transformed_image = transformed_image.reshape(-1, 3, self.output_size[0], self.output_size[1])

        x['images'] = transformed_image
        return x
    

# TODO: document this
class ViewpointTransformation():
    ROTATIONS = [0, 90, 270, 180]

    def __init__(self, target_size, crop_size, downsampling_factor, rotation_angles=[0], add_flipped_viewport=False):
        self.target_size = target_size
        self.crop_size = crop_size
        self.downsampling_factor = downsampling_factor
        self.rotation_angles = rotation_angles
        self.add_flipped_viewport = add_flipped_viewport

    def __call__(self, x):
        transformed_image = x['images']
        transformed_image = transformed_image.reshape(-1,transformed_image.shape[-3],transformed_image.shape[-2],transformed_image.shape[-1])
        result = torch.zeros((4*len(self.rotation_angles)* # number rotations
                             (2 if self.add_flipped_viewport else 1), # with flipping?
                             transformed_image.shape[0], # number images
                             3,self.target_size[0],self.target_size[1]))    #(16,16,3,4,45)
        
        n = 0
        for angle in self.rotation_angles:
            rotation  = TF.rotate(transformed_image, angle)
            crop = TF.center_crop(rotation, [int(self.downsampling_factor * i) for i in self.crop_size])   #69*3
            resize = TF.resize(crop, self.crop_size, antialias=True)
            for f in range(2 if self.add_flipped_viewport else 1):
                if f==1:
                    resize = TF.hflip(resize)
                four_crop = TF.five_crop(resize, self.target_size)[:-1] # ignor the center crop -1
                for i in range (len(four_crop)):
                    #new_x = copy.copy(x) # non flipped crop
                    #new_x['image'] 
                    result[n] = TF.rotate(four_crop[i], ViewpointTransformation.ROTATIONS[i])
                    n = n+1
                    #result.append(new_x)
        result = result.swapaxes(0,1).reshape(-1,3,self.target_size[0],self.target_size[1])
        x['images'] = result
        return x
"""
For each test image, the predictions for these 60 affine transformations are uniformly averaged.
Since we are only using the best model, the additional transformations are not required.

class AffineTestTransformation():
    def __init__(self, rotation_angles=[0], scalings=[1], flip=True):
        self.rotation_angles = rotation_angles
        self.scalings = scalings
        self.flip = flip

    def __call__(self, x):
        images = x['images']
        result = torch.zeros((len(self.rotation_angles) * len(self.scalings) * (2 if self.flip else 1), 3, images.shape[-2], images.shape[-1]))
        n = 0
        for angle in self.rotation_angles:
            rotation = TF.rotate(images, angle)
            for scale_factor in self.scalings:
                scaled_image = TF.affine(rotation, angle=0, translate=[0, 0], scale=scale_factor, shear=0)
                
                # Append original scaled image
                result[n] = scaled_image
                n += 1
                
                # Check if flipping is enabled
                if self.flip:
                    # Apply horizontal flipping
                    flipped_image = TF.hflip(scaled_image)
                    
                    # Append flipped image
                    result[n] = flipped_image
                    n += 1

        x['images'] = result
        return x

"""