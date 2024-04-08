import DataSets
import Preprocessing

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

class AlexNetTransferLearning(LightningModule):
    def __init__(self):
        super().__init__()
        
        #initializing Alexnet feature extractor
        backbone = models.alexnet(weights="DEFAULT")
        self.feature_extractor = backbone.features
        self.feature_extractor.eval()
        
        #initializing Dieleman classifier
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.001)
        nn.init.constant_(self.fc1.bias, 0.01)

        self.maxpool4 = nn.MaxPool1d(kernel_size=2)

        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=2048, out_features=4096)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.001)
        nn.init.constant_(self.fc2.bias, 0.01)

        self.maxpool5 = nn.MaxPool1d(kernel_size=2)

        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=2048, out_features=37)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc3.bias, 0.1)


        self.batchsize=256
        self.gradient_steps = 0    # Counter for gradient steps
        self.relu = nn.ReLU()

        #initializing the variables needed for divisive normalisation
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

    #defining the methods needed for divisive normalisation
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
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.dropout1(representations)
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

        # Use the first 99% of the images for training, and the last 1% for validation
        self.train_indices = list(range(0, split_point))
        self.val_indices = list(range(split_point, num_images))

    def train_dataloader(self):
        '''
        This function returns the training dataloader with preprocessed data.
        '''
        train_dataset = DataSets.GalaxyZooDataset(data_directory = '/local_data/AIN/Renuka/KaggleGalaxyZoo/images_training_rev1',
                                                     label_file = '../data/KaggleGalaxyZoo/training_solutions_rev1.csv',
                                                     extension = '.jpg',
                                                     transform = transforms.Compose([Preprocessing.AlexnetTransformation(resize=256, 
                                                                                                     centercrop=224, 
                                                                                                     mean=[0.485, 0.456, 0.406], 
                                                                                                     std=[0.229, 0.224, 0.225]),
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
        validation_dataset = DataSets.GalaxyZooDataset(data_directory = '/local_data/AIN/Renuka/KaggleGalaxyZoo/images_training_rev1',
                                                     label_file = '../data/KaggleGalaxyZoo/training_solutions_rev1.csv',
                                                     extension = '.jpg',
                                                     transform = transforms.Compose([Preprocessing.AlexnetTransformation(resize=256, 
                                                                                                     centercrop=224, 
                                                                                                     mean=[0.485, 0.456, 0.406], 
                                                                                                     std=[0.229, 0.224, 0.225]),
                                                                                     Preprocessing.ViewpointTransformation(target_size=[45,45], 
                                                                                                                           crop_size=[69,69], 
                                                                                                                           downsampling_factor=3.0, 
                                                                                                                           rotation_angles=[0,45], 
                                                                                                                           add_flipped_viewport=True)]))
        
        val_dataset = Subset(validation_dataset, self.val_indices)
        
        validation_dataloader = DataLoader( val_dataset,
                                            batch_size=self.batchsize,
                                            shuffle=False,
                                            num_workers=48)
        return validation_dataloader


    def test_dataloader(self):
        testing_dataset = DataSets.GalaxyZooDataset(data_directory = '/local_data/AIN/Renuka/KaggleGalaxyZoo/images_test_rev1',
                                                    extension = '.jpg',
                                                    transform = transforms.Compose([Preprocessing.AlexnetTransformation(resize=256, 
                                                                                                    centercrop=224, 
                                                                                                    mean=[0.485, 0.456, 0.406], 
                                                                                                    std=[0.229, 0.224, 0.225]),
                                                                                    Preprocessing.ViewpointTransformation(target_size=[45,45], 
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
        scheduler = MultiStepLR(optimizer, milestones=[40, 85], gamma=0.1) 
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        '''
        This function performs a single training step.
        '''
        views = batch['images'].reshape(-1,3,224,224)
        # Disabling normaliser for the first 625 gradient steps as per Dieleman's paper
        if self.gradient_steps < 625:
            output = self.forward(views.to('cuda'))
        else:
            output = self.normaliser(views.to('cuda'))

        rmse = torch.sqrt(torch.mean(torch.square(output - batch['labels'].to('cuda')), axis=1))
        loss = torch.mean(rmse)   #average RMSE loss across the batch
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
       
        # Incrementing the gradient steps counter
        self.gradient_steps += 1
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        '''
        This function performs a single validation step.
        '''
        views = batch['images'].reshape(-1,3,224,224)
        output = self.normaliser(views.to('cuda'))
        rmse = torch.sqrt(torch.mean(torch.square(output - batch['labels'].to('cuda')), axis=1))
        loss = torch.mean(rmse)
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        '''
        This function performs a single testing step.
        '''
        views = batch['images'].reshape(-1,3,224,224)
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
    dirpath='alexnet_dielemann_16',
    filename='model-{epoch:02d}-{train_loss:.2f}',
    save_top_k=1,
    mode='min',
    save_last=True
)   


    
#testing  
# if __name__ == '__main__':
#     pl.seed_everything(123456)
#     network = AlexNetTransferLearning.load_from_checkpoint(checkpoint_path="alexnet_dieleman_fc/model-epoch=46-train_loss=0.08.ckpt",
#                                                   hparams_file="lightning_logs/version_0/hparams.yaml",
#     )
#     trainer = Trainer(devices=1, accelerator="gpu")
#     trainer.test(network, verbose=True) 

#training
if __name__ == '__main__':
    pl.seed_everything(123456)
    network = AlexNetTransferLearning()
    trainer = Trainer(max_epochs=90, devices=1, accelerator="gpu", callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback])
    trainer.fit(network)
    

    
