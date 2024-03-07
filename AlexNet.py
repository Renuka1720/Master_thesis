import DataSets

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR
torch.autograd.set_detect_anomaly(True)

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os

# TODO: add documentation
class AlexNetModel(LightningModule):
    def __init__(self):
        super(AlexNetModel, self).__init__()

        # Load pretrained AlexNet model
        self.model = models.alexnet(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_classes = 37
        # Modify the last fully connected layer for your number of classes
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        # Unfreeze the last layer
        for param in self.model.classifier[6].parameters():
            param.requires_grad = True

        self.batchsize=16

        # # Rectification non-linearity (ReLU)
        # self.relu = nn.ReLU()
        self.question_slices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9), slice(9, 13), slice(13, 15),
                                slice(15, 18), slice(18, 25), slice(25, 28), slice(28, 31), slice(31, 37)]

        self.normalisation_mask = self.generate_normalisation_mask()

        self.scaling_sequence = [
            (slice(3, 5), 1),
            (slice(5, 13), 4),
            (slice(15, 18), 0),
            (slice(18, 25), 13),
            (slice(25, 28), 3),
            (slice(28, 37), 7),
        ]

    def generate_normalisation_mask(self):
        mask = torch.zeros(37, 37).to('cuda')
        for s in self.question_slices:
            mask[s, s] = 1.0
        return mask

    def calculate_normalized_outputs(self, x):
        x_clipped = self.relu(x) # negative values become 0
        normalisation_denoms = (torch.mm(x_clipped, self.normalisation_mask)) + 1e-12
        x_normalised = x_clipped / normalisation_denoms
        # x_normalised_clone = x_normalised.clone()
        for probs_slice, scale_idx in self.scaling_sequence:
            x_normalised[:, probs_slice] = x_normalised[:, probs_slice] * x_normalised[:, scale_idx].unsqueeze(1)

        return x_normalised

    # TODO: check if the input image is a tensor
    def forward(self, x):
        return self.model(x)

    def normaliser(self, x):
        return self.calculate_normalized_outputs(self.forward(x))

    def prepare_data(self):
        full_dataset = DataSets.GalaxyZooDataset(
            data_directory='../data/KaggleGalaxyZoo/images_training_rev1',
            label_file='../data/KaggleGalaxyZoo/training_solutions_rev1.csv',
            extension='.jpg',
        )

        num_images = len(full_dataset)
        split_point = int(0.99 * num_images)  # Point to split the dataset

        # Use the first 90% of the images for training, and the last 10% for validation
        self.train_indices = list(range(0, split_point))
        self.val_indices = list(range(split_point, num_images))

    def train_dataloader(self):
        train_dataset = DataSets.GalaxyZooDataset(data_directory = '/local_data/AIN/Renuka/KaggleGalaxyZoo/images_training_rev1',
                                                     label_file = '../data/KaggleGalaxyZoo/training_solutions_rev1.csv',
                                                     extension = '.jpg',
                                                     transform = transforms.Compose([   transforms.Resize(256),                    
                                                                                        transforms.CenterCrop(224),                
                                                                                        transforms.ToTensor(),                     
                                                                                        transforms.Normalize(                      
                                                                                        mean=[0.485, 0.456, 0.406],                
                                                                                        std=[0.229, 0.224, 0.225]                  
                                                                                        )]))
                                                    #  transform = transforms.Compose([Preprocessing.DielemanTransformation(rotation_range=[0,360], 
                                                    #                                                                       translation_range=[4./424,4./424], 
                                                    #                                                                       scaling_range=[1/1.3,1.3], 
                                                    #                                                                       flip=0.5),
                                                    #                                  Preprocessing.KrizhevskyColorTransformation(weights=[-0.0148366, -0.01253134, -0.01040762], 
                                                    #                                                                              std=0.5),
                                                    #                                  Preprocessing.ViewpointTransformation(target_size=[45,45], 
                                                    #                                                                        crop_size=[69,69], 
                                                    #                                                                        downsampling_factor=3.0, 
                                                    #                                                                        rotation_angles=[0,45], 
                                                    #                                                                        add_flipped_viewport=True)]))
        '''
        # Calculate the number of samples for validation
        no_of_samples = len(training_dataset)
        val_samples = int(0.1 * no_of_samples)  # 10 percent of the training data
        train_samples = no_of_samples - val_samples

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(training_dataset, [train_samples, val_samples])
        '''
        training_dataset = Subset(train_dataset, self.train_indices)
        
        training_dataloader = DataLoader(training_dataset,
                                      batch_size=self.batchsize,
                                      shuffle=True,
                                      num_workers=48)
        return training_dataloader

    
    def val_dataloader(self):
        validation_dataset = DataSets.GalaxyZooDataset(data_directory = '../data/KaggleGalaxyZoo/images_training_rev1',
                                                     label_file = '../data/KaggleGalaxyZoo/training_solutions_rev1.csv',
                                                     extension = '.jpg',
                                                     transform = transforms.Compose([   transforms.Resize(256),                    
                                                                                        transforms.CenterCrop(224),                
                                                                                        transforms.ToTensor(),                     
                                                                                        transforms.Normalize(                      
                                                                                        mean=[0.485, 0.456, 0.406],                
                                                                                        std=[0.229, 0.224, 0.225]                  
                                                                                        )]))
                                                    #  transform = transforms.Compose([Preprocessing.ViewpointTransformation(target_size=[45,45], crop_size=[69,69], downsampling_factor=3.0, rotation_angles=[0,45], add_flipped_viewport=True)]))

        # # Calculate the number of samples for validation
        # no_of_samples = len(training_dataset)
        # val_samples = int(0.1 * no_of_samples)  # 1
        # train_samples = no_of_samples - val_samples

        # # Split the dataset into training and validation sets
        # train_dataset, val_dataset = random_split(training_dataset, [train_samples, val_samples])
        
        val_dataset = Subset(validation_dataset, self.val_indices)
        validation_dataloader = DataLoader(val_dataset,
                                    batch_size=self.batchsize,
                                    shuffle=False,
                                    num_workers=48)
        return validation_dataloader


    def test_dataloader(self):
        testing_dataset = DataSets.GalaxyZooDataset(data_directory = '/local_data/AIN/Renuka/KaggleGalaxyZoo/images_test_rev1',
                                                    extension = '.jpg',
                                                    transform = transforms.Compose([   transforms.Resize(256),                    
                                                                                        transforms.CenterCrop(224),                
                                                                                        transforms.ToTensor(),                     
                                                                                        transforms.Normalize(                      
                                                                                        mean=[0.485, 0.456, 0.406],                
                                                                                        std=[0.229, 0.224, 0.225]                  
                                                                                        )]))
                                                    # transform = transforms.Compose([
                                                    #     # Preprocessing.AffineTestTransformation(rotation_angles=[0, 36, 72, 108, 144, 180, 216, 252, 288, 324],
                                                    #     #                                                                    scalings=[1/1.2,1.0,1.2],
                                                    #     #                                                                    flip=True),
                                                    #                                 Preprocessing.ViewpointTransformation(target_size=[45,45], 
                                                    #                                                                       crop_size=[69,69], 
                                                    #                                                                       downsampling_factor=3.0, 
                                                    #                                                                       rotation_angles=[0,45], 
                                                    #                                                                       add_flipped_viewport=True)]))
        testing_dataloader = DataLoader(testing_dataset,
                                        batch_size=self.batchsize,
                                        num_workers=48,
                                        shuffle=False)
        return testing_dataloader
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.04, momentum=0.9, nesterov=True)
        scheduler = MultiStepLR(optimizer, milestones=[286, 364], gamma=0.1) # 18mio images and 23mio images [325, 415]
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        # self.use_dropout = True
        views = batch['images'].reshape(-1,3,45,45) #first 16 views of 1 image .. 16 views of 2 image and so on
        # Disable normaliser for the first 625 gradient steps
        if self.gradient_steps < 625:
            output = self.forward(views.to('cuda'))
        else:
            output = self.normaliser(views.to('cuda'))
        #output = self.forward(views.to('cuda'))
        rmse = torch.sqrt(torch.mean(torch.square(output - batch['labels'].to('cuda')), axis=1))
        loss = torch.mean(rmse)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
       
        # Increment the gradient steps counter
        #self.gradient_steps += 1
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
    # Reshape views
        views = batch['images'].reshape(-1,3,45,45)

        # Forward pass
        output = self.normaliser(views.to('cuda'))
        #TODO: change it back to self.normaliser after checking without div normalisation

        # Compute RMSE
        rmse = torch.sqrt(torch.mean(torch.square(output - batch['labels'].to('cuda')), axis=1))

        # Compute loss
        loss = torch.mean(rmse)

        # Log the validation loss
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)

        return {'val_loss': loss}

    
    #test_set = set()
    def test_step(self, batch, batch_idx):
        # self.use_dropout = False
        #test_set.add(batch_idx)
        # if batch_idx <4995:
        #     return None
        views = batch['images'].reshape(-1,3,45,45)
        output = self.normaliser(views.to('cuda'))
        #output = output.reshape(int(views.shape[0]/60/self.viewpoints), 60, 37)
        #output = torch.mean(output, axis=1)
        output = torch.clip(output, min=0, max=1)

        if not os.path.exists('results.csv'):
            with open('results.csv', 'w') as file:
                file.write('GalaxyID,Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,Class5.1,Class5.2,Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,Class10.2,Class10.3,Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,Class11.6\n')

        with open('results.csv', 'a') as file:
            for i in range(output.shape[0]):
                file.write(str(int(batch['id'][i].cpu().numpy()))+",")
                for t in range(output.shape[1]):
                    file.write(str(output[i][t].cpu().numpy()))
                    if t < output.shape[1]-1:
                        file.write(",")
                    else:
                        file.write("\n")


checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',
    dirpath='AlexNet_model/',
    filename='model-{epoch:02d}-{train_loss:.2f}',
    save_top_k=1,
    mode='min',
    save_last=True
)   


    
#testing  
# if __name__ == '__main__':
#     pl.seed_everything(123456)
#     network = DielemannModel.load_from_checkpoint(checkpoint_path="AlexNet_model/last.ckpt",
#                                                   hparams_file="lightning_logs/version_11/hparams.yaml",
#     )
#     trainer = Trainer(devices=1, accelerator="gpu")
#     trainer.test(network, verbose=True) 

#training
if __name__ == '__main__':
    pl.seed_everything(123456)
    network = AlexNetModel()#.load_from_checkpoint(checkpoint_path="val_model/model-epoch=368-train_loss=0.07.ckpt")
    trainer = Trainer(max_epochs=100, devices=1, accelerator="gpu", callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback]) #dielemann-> 452 epochs
    trainer.fit(network)