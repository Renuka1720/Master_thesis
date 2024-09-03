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


class block(LightningModule):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(LightningModule):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
        
        self.batchsize=16
        self.viewpoints=16
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


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
       # a, b = next(iter(training_dataloader))
       # print(a.shape, b.shape)
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
    dirpath='scratch_resnet',
    filename='model-{epoch:02d}-{train_loss:.2f}',
    save_top_k=1,
    mode='min',
    save_last=True
)   

def ResNet50(img_channel=3, num_classes=37):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)
    
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #global seed for reproducibility of results across multiple function calls
    pl.seed_everything(123456)
    # instance of the DielemannModel
    network = ResNet50(img_channel=3, num_classes=37)#.load_from_checkpoint(checkpoint_path="val_model/model-epoch=368-train_loss=0.07.ckpt")
    # Initializing the trainer object
    trainer = Trainer(max_epochs=500, devices=1, accelerator="gpu", callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback])
    #Starting the training process using the Trainer
    trainer.fit(network)
