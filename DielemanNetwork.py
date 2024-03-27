import DataSets
import Preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
torch.autograd.set_detect_anomaly(True)

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os

# TODO: add documentation
class DielemannModel(LightningModule):
    def __init__(self):
        super(DielemannModel, self).__init__()

        self.batchsize=16
        self.viewpoints=16
        #Output size after convolution filter = ((w-f+2P)/s) +1
        self.gradient_steps = 0    # Counter for gradient steps

        #Input shape= (16,3,45,45)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=6)
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv1.bias, 0.1)

        # Max-pooling layer
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Shape= (16,6,40,40)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv2.bias, 0.1)

        # Max-pooling layer
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Convolutional layers
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv3.bias, 0.1)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.1)
        nn.init.constant_(self.conv4.bias, 0.1)

        # Max-pooling layer
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # Fully connected layers (or) dense layers
        # self.use_dropout = False

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features= 8192, out_features= 4096)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.001)
        nn.init.constant_(self.fc1.bias, 0.01)

        self.maxpool4 = nn.MaxPool1d(kernel_size=2)

        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features= 2048, out_features= 4096)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.001)
        nn.init.constant_(self.fc2.bias, 0.01)

        self.maxpool5 = nn.MaxPool1d(kernel_size=2)

        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features= 2048, out_features= 37)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc3.bias, 0.1)

        # Rectification non-linearity (ReLU)
        self.relu = nn.ReLU()
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
        x_normalised_clone = x_normalised.clone()
        x_normalised_clone[:, slice(3,5)] = x_normalised[:, slice(3,5)] * x_normalised[:, 1].unsqueeze(1)
        x_normalised_clone[:, slice(5,13)] = x_normalised[:, slice(5,13)] * x_normalised[:, 1].unsqueeze(1) * x_normalised[:, 4].unsqueeze(1)
        x_normalised_clone[:, slice(15,18)] = x_normalised[:, slice(15,18)] * x_normalised[:, 0].unsqueeze(1)
        x_normalised_clone[:, slice(18,25)] = x_normalised[:, slice(18,25)] * x_normalised[:, 13].unsqueeze(1)
        x_normalised_clone[:, slice(25,28)] = x_normalised[:, slice(25,28)] * x_normalised[:, 1].unsqueeze(1) * x_normalised[:, 3].unsqueeze(1)
        x_normalised_clone[:, slice(28,37)] = x_normalised[:, slice(28,37)] * x_normalised[:, 1].unsqueeze(1) * x_normalised[:, 4].unsqueeze(1) * x_normalised[:, 7].unsqueeze(1)
        return x_normalised_clone

    def forward(self, x):
        # Convolutional layers Shape= (batchsize*16,3,45,45)
        x = self.conv1(x)  # Shape= (batchsize*16,32,40,40)
        x = self.relu(x)
        x = self.maxpool1(x) #Shape= (batchsize*16,32,20,20)

        x = self.conv2(x)  #Shape= (batchsize*16,64,16,16)
        x = self.relu(x)
        x = self.maxpool2(x)  #Shape= (batchsize*16,64,8,8)

        x = self.conv3(x)    #Shape= (batchsize*16,128,6,6)
        x = self.relu(x)

        x = self.conv4(x)   #Shape= (batchsize*16,128,4,4)
        x = self.relu(x)
        x = self.maxpool3(x)  #Shape= (batchsize*16,128,2,2)

        # Flattening the feature maps
        x = x.view(-1, self.viewpoints*512)   #2d tensor of shape= (batchsize, 16*128*2*2 = 512*16=8192)

        # Fully connected layers
        # Dropout is disabled by default in PyTorch Lightning during testing. Therefore, it is not necessary to manually turn it off each time you test.

        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.maxpool4(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.maxpool5(x)

        x = self.dropout3(x)
        x = self.fc3(x)

        return x

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

        # Using the first 90% of the images for training, and the last 10% for validation
        self.train_indices = list(range(0, split_point))
        self.val_indices = list(range(split_point, num_images))

    def train_dataloader(self):
        '''
        This function returns the training dataloader with preprocessed data.
        '''
        # Defining the train dataset with preprocessing transformations
        train_dataset = DataSets.GalaxyZooDataset(data_directory = '/local_data/AIN/Renuka/KaggleGalaxyZoo/images_training_rev1',
                                                     label_file = '../data/KaggleGalaxyZoo/training_solutions_rev1.csv',
                                                     extension = '.jpg',
                                                     transform = transforms.Compose([Preprocessing.DielemanTransformation(rotation_range=[0,360], 
                                                                                                                          translation_range=[4./424,4./424], 
                                                                                                                          scaling_range=[1/1.3,1.3], 
                                                                                                                          flip=0.5),
                                                                                     Preprocessing.KrizhevskyColorTransformation(weights=[-0.0148366, -0.01253134, -0.01040762], 
                                                                                                                                 std=0.5),
                                                                                     Preprocessing.ViewpointTransformation(target_size=[45,45], 
                                                                                                                           crop_size=[69,69], 
                                                                                                                           downsampling_factor=3.0, 
                                                                                                                           rotation_angles=[0,45], 
                                                                                                                           add_flipped_viewport=True)]))
        
        # training data subset
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
                                                     transform = transforms.Compose([Preprocessing.ViewpointTransformation(target_size=[45,45], crop_size=[69,69], downsampling_factor=3.0, rotation_angles=[0,45], add_flipped_viewport=True)]))
        
        val_dataset = Subset(validation_dataset, self.val_indices)
        validation_dataloader = DataLoader(val_dataset,
                                    batch_size=self.batchsize,
                                    shuffle=False,
                                    num_workers=48)
        return validation_dataloader


    def test_dataloader(self):
        testing_dataset = DataSets.GalaxyZooDataset(data_directory = '/local_data/AIN/Renuka/KaggleGalaxyZoo/images_test_rev1',
                                                    extension = '.jpg',
                                                    transform = transforms.Compose([Preprocessing.ViewpointTransformation(target_size=[45,45], 
                                                                                                                          crop_size=[69,69], 
                                                                                                                          downsampling_factor=3.0, 
                                                                                                                          rotation_angles=[0,45], 
                                                                                                                          add_flipped_viewport=True)]))
        testing_dataloader = DataLoader(testing_dataset,
                                        batch_size=self.batchsize,
                                        num_workers=48,
                                        shuffle=False)
        return testing_dataloader
    
    
    def configure_optimizers(self):
        '''
        This function sets up the optimizer and learning rate scheduler.
        '''
        optimizer = torch.optim.SGD(self.parameters(), lr=0.04, momentum=0.9, nesterov=True)
        scheduler = MultiStepLR(optimizer, milestones=[286, 364], gamma=0.1) # 18mio images and 23mio images [325, 415]
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        '''
        This function performs a single training step.
        '''
        views = batch['images'].reshape(-1,3,45,45) #first 16 views of 1 image .. 16 views of 2 image and so on
        # Forward pass
        # Disable normaliser for the first 625 gradient steps
        if self.gradient_steps < 625:
            output = self.forward(views.to('cuda'))
        else:
            output = self.normaliser(views.to('cuda'))
        
        #Computing RMSE for each sample in the batch
        rmse = torch.sqrt(torch.mean(torch.square(output - batch['labels'].to('cuda')), axis=1))
        #computes the mean of the RMSE values across the batch, resulting in a single scalar value representing the average RMSE loss across the batch
        loss = torch.mean(rmse)
        #logging the training loss
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
       
        # Increment the gradient steps counter
        self.gradient_steps += 1
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        '''
        This function performs a single validation step.
        '''
    # Reshaping views
        views = batch['images'].reshape(-1,3,45,45)

        # Forward pass
        output = self.normaliser(views.to('cuda'))
        # Computing RMSE
        rmse = torch.sqrt(torch.mean(torch.square(output - batch['labels'].to('cuda')), axis=1))
        # loss
        loss = torch.mean(rmse)

        # Log the validation loss
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)

        return {'val_loss': loss}

    
    def test_step(self, batch, batch_idx):
        '''
        This function performs a single testing step.
        '''
        views = batch['images'].reshape(-1,3,45,45)
        output = self.normaliser(views.to('cuda'))
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
    dirpath='Dieleman_check',
    filename='model-{epoch:02d}-{train_loss:.2f}',
    save_top_k=1,
    mode='min',
    save_last=True
)   


    
#testing  
# if __name__ == '__main__':
#     pl.seed_everything(123456)
#     network = DielemannModel.load_from_checkpoint(checkpoint_path="Dieleman_check/model-epoch=392-train_loss=0.02.ckpt",
#                                                   hparams_file="lightning_logs/version_18/hparams.yaml",
#     )
#     trainer = Trainer(devices=1, accelerator="gpu")
#     trainer.test(network, verbose=True) 

#training
if __name__ == '__main__':
    #global seed for reproducibility of results across multiple function calls
    pl.seed_everything(123456)
    # instance of the DielemannModel
    network = DielemannModel()#.load_from_checkpoint(checkpoint_path="val_model/model-epoch=368-train_loss=0.07.ckpt")
    # Initializing the trainer object
    trainer = Trainer(max_epochs=406, devices=1, accelerator="gpu", callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback])
    #Starting the training process using the Trainer
    trainer.fit(network)