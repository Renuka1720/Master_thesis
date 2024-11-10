"""
irrep to trivial in the last conv layer for invariance
Change no of conv and fc layers, features according to the problem in hand
can increase BS to utilise full GPU
check warnings
AdamW 
Learningrate finder or Optuna
"""

import DataSets1
import Preprocessing

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms.functional as TF


import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, LearningRateFinder, Callback

#from optuna.integration import PyTorchLightningPruningCallback


import e2cnn
from e2cnn import gspaces
from e2cnn import nn

import math
import os
import time

# import torch.nn.functional as F
# x =F.softmax(x,dim=1)

# import torch.nn
# self.softmax = nn.softmax(dim=1)
early_stopping_callback = EarlyStopping(
                                        monitor='val_loss',  # Monitor the validation loss
                                        patience=7,          # Number of epochs with no improvement after which training will be stopped
                                        verbose=True,        # Prints a message if early stopping is triggered
                                        mode='min'           # The monitored metric should be minimized (as we want to minimize loss)
                                    )


class C8SteerableCNN(LightningModule):
    
    def __init__(self, n_classes=37):
        
        super(C8SteerableCNN, self).__init__()
        
        self.batchsize = 272 

        # the model is equivariant under rotations and reflections, modelled by C16
        self.r2_act = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency= 7)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, 3* [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the irreps representation of C8
        out_type = nn.FieldType(self.r2_act, 24*[self.r2_act.irrep(0,0)])
        self.block1 = nn.SequentialModule(
            nn.MaskModule(in_type, 60, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.GNormBatchNorm(out_type),
            nn.NormNonLinearity(out_type)
        )
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 irreps feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.irrep(1,2)])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.GNormBatchNorm(out_type),
            nn.NormNonLinearity(out_type)
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)  

        )
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 irreps feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.irrep(0,0)])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.GNormBatchNorm(out_type),
            nn.NormNonLinearity(out_type)
        )
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 irreps feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.irrep(0,0)])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.GNormBatchNorm(out_type),
            nn.NormNonLinearity(out_type)
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 irreps feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.irrep(0,0)])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.GNormBatchNorm(out_type),
            nn.NormNonLinearity(out_type)
        )
        
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 irreps feature fields of C8
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.trivial_repr])  #[self.r2_act.irrep(1,2)])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.GNormBatchNorm(out_type),
            nn.NormNonLinearity(out_type)
        )
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        
        self.gpool = nn.GroupPooling(out_type)
        
        # number of output channels
        c = self.gpool.out_type.size
        
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(4096, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
        )
    
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)
        
        # apply each equivariant block
        
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        
        x = self.block5(x)
        x = self.block6(x)
        
        # pool over the spatial dimensions
        x = self.pool3(x)
        
        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
        
        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        return x
    
    def prepare_data(self):
        # full_dataset = DataSets1.GalaxyZooDataset(
        #     data_directory='/local_data/AIN/Renuka/KaggleGalaxyZoo/images_training_rev1',
        #     label_file='/local_data/AIN/Renuka/KaggleGalaxyZoo/training_solutions_rev1.csv',
        #     extension='.jpg',
        # )

        # num_images = len(full_dataset)
        num_images = 61578
        split_point = int(math.floor((0.98 * num_images)))  # Point to split the dataset

        # Use the first 99% of the images for training, and the last 1% for validation
        self.train_indices = list(range(0, split_point))
        self.val_indices = list(range(split_point, num_images))
        print(f"Train samples: {len(self.train_indices)}")
        print(f"Validation samples: {len(self.val_indices)}")

    def train_dataloader(self):
        '''
        This function returns the training dataloader with preprocessed data.
        '''
        print('Preparing training data...')
        train_dataset = DataSets1.GalaxyZooDataset(data_directory = '/local_data/AIN/Renuka/KaggleGalaxyZoo/cropped_training',
                                                     label_file = '/local_data/AIN/Renuka/KaggleGalaxyZoo/training_solutions_rev1.csv',
                                                     transform = transforms.Compose([Preprocessing.KrizhevskyColorTransformation(weights=[-0.0148366, -0.01253134, -0.01040762], 
                                                                                                                                 std=0.5),
                                                                                     Preprocessing.NoViewpointTransformation(output_size= [60,60], downsampling_factor=3.0)])) 
        
        print('After Preparing training data...')                                                                                  
        # training data subset
        training_dataset = Subset(train_dataset, self.train_indices)
        print('After whatever this is') 
        
        training_dataloader = DataLoader(training_dataset,
                                        batch_size=self.batchsize,
                                        shuffle=True,
                                        num_workers=48, 
                                        persistent_workers=True)
        print('Hi') 
        return training_dataloader

    
    def val_dataloader(self):
        validation_dataset = DataSets1.GalaxyZooDataset(data_directory = '/local_data/AIN/Renuka/KaggleGalaxyZoo/cropped_training',
                                                     label_file = '/local_data/AIN/Renuka/KaggleGalaxyZoo/training_solutions_rev1.csv',
                                                     #extension = '.jpg',
                                                     transform = transforms.Compose([Preprocessing.KrizhevskyColorTransformation(weights=[-0.0148366, -0.01253134, -0.01040762], 
                                                                                                                                 std=0.5),
                                                                                     Preprocessing.NoViewpointTransformation(output_size= [60,60], downsampling_factor=3.0)])) 
        

        val_dataset = Subset(validation_dataset, self.val_indices)
        
        validation_dataloader = DataLoader( val_dataset,
                                            batch_size=self.batchsize,
                                            shuffle=False,
                                            num_workers=48)
        return validation_dataloader


    def test_dataloader(self):
        testing_dataset = DataSets1.GalaxyZooDataset(data_directory = '/local_data/AIN/Renuka/KaggleGalaxyZoo/images_test_rev1',
                                                    #extension = '.jpg',
                                                    transform = transforms.Compose([Preprocessing.NoViewpointTransformation(output_size= [60,60], downsampling_factor=3.0)]))
      
        testing_dataloader = DataLoader(testing_dataset,
                                        batch_size=self.batchsize,
                                        num_workers=8,
                                        shuffle=False)
        return testing_dataloader
    
    
    def configure_optimizers(self):
        '''
        This function sets up the optimizer and learning rate scheduler.
        '''
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, nesterov=True)
        scheduler = MultiStepLR(optimizer, milestones=[2000, 36000], gamma=0.1) # 18mio images and 23mio images [325, 415]
        #scheduler = MultiStepLR(optimizer, milestones=[30, 65], gamma=0.1) 
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        '''
        This function performs a single training step.
        '''
        views = batch['images'].reshape(-1,3,60,60)
        # Disabling normaliser for the first 625 gradient steps as per Dieleman's paper
        
        output = self.forward(views.to('cuda'))
        rmse = torch.sqrt(torch.mean(torch.square(output - batch['labels'].to('cuda')), axis=1))
        loss = torch.mean(rmse)   #average RMSE loss across the batch
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size= self.batchsize)
       
        # Incrementing the gradient steps counter
        #self.gradient_steps += 1
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        '''
        This function performs a single validation step.
        '''
        views = batch['images'].reshape(-1,3,60,60) #(-1,3,45,45)
        output = self.forward(views.to('cuda'))
        rmse = torch.sqrt(torch.mean(torch.square(output - batch['labels'].to('cuda')), axis=1))
        loss = torch.mean(rmse)
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True, batch_size= self.batchsize)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        '''
        This function performs a single testing step.
        '''
        views = batch['images'].reshape(-1,3,60,60)
        output = self.forward(views.to('cuda'))
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
    monitor='val_loss',
    dirpath='/local_data/AIN/Renuka/checkpoints/e2cnn_no_VP',
    filename='model-{epoch:02d}-{train_loss:.2f}',
    save_top_k=1,
    every_n_epochs= 3,
    mode='min',
    save_last=True
)   

wandb_logger = WandbLogger(project="e2cnn")

    
#testing  
# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     pl.seed_everything(123456)
#     network = ResNetTransferLearning.load_from_checkpoint(checkpoint_path="/local_data/AIN/Renuka/checkpoints/resnet_no_VP/model-epoch=1799-train_loss=0.07.ckpt",
#     )
#     #network.freeze()
#     trainer = Trainer(devices=1, accelerator="gpu")
#     trainer.test(network, verbose=True) 

#training
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    pl.seed_everything(123456)
    network = C8SteerableCNN()
    trainer = Trainer(
            max_epochs=50000, 
            devices= 1,
            accelerator="gpu", 
            check_val_every_n_epoch=1, 
            logger=wandb_logger,
            callbacks=[LearningRateMonitor(logging_interval='step'),
            checkpoint_callback, early_stopping_callback]
                        )
    trainer.fit(network) 