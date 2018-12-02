import torch

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from data_load import FacialKeypointsDataset,FaceLandmarksDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor

kwargs = {'num_workers': 4} if torch.cuda.is_available() else {}

# define the data_transform


data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])
'''


data_dir='/mnt/data2/niceliu/multi_task_facial_landmark_datatset'
train_dataset=FaceLandmarksDataset(root_dir=data_dir,
                                           filelist='training.txt',
                                           transform=transform_train)
batch_size = 256

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          **kwargs)


test_dataset = FaceLandmarksDataset(root_dir=data_dir,
                                           filelist='testing.txt',
                                           transform=transform_train)

batch_size = 128

test_loader = DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          **kwargs)

'''
# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                             root_dir='./data/training/',
                                             transform=data_transform)

# load training data in batches
batch_size = 128

train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          **kwargs)

# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='./data/test_frames_keypoints.csv',
                                             root_dir='./data/test/',
                                             transform=data_transform)

# load test data in batches
#batch_size = 10

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=False,
                          **kwargs)


# test the model on a batch of test images
def net_sample_output(net, device):
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):

        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # wrap images in a torch Variable
        # key_pts do not need to be wrapped until they are used for training
        images = Variable(images)

        # convert images to FloatTensors
        if (torch.cuda.is_available()):
            images = images.type(torch.cuda.FloatTensor)
            images.to(device)
        else:
            images = images.type(torch.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)

        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], -1, 2)

        # break after first image is tested
        if i == 1:
            return images, output_pts, key_pts


import visdom
import numpy as np
def train_net(net, device, criterion, optimizer, n_epochs):

    num_iter=0
    val_iter = 0
    vis=visdom.Visdom()
    win=vis.line(Y=np.array([0]),X=np.array([0]))
    win_=vis.line(Y=np.array([0]),X=np.array([0]))
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)
            
            # wrap them in a torch Variable
            images, key_pts = Variable(images), Variable(key_pts)

            # convert variables to floats for regression loss
            if (torch.cuda.is_available()):
                key_pts = key_pts.type(torch.cuda.FloatTensor)
                images = images.type(torch.cuda.FloatTensor)
                images.to(device)
            else:
                key_pts = key_pts.type(torch.FloatTensor)
                images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)
            running_loss += loss.item()
            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            if (batch_i+1) % 10 == 0:
                print('Epoch [{}/{}],Iter [{}/{}] Loss: {:.4f}, average_loss: {:.4f}'.format(
                    epoch+1,n_epochs,batch_i+1,len(train_loader),loss.item(),running_loss/(batch_i+1)))

                vis.line(Y=np.array([running_loss/(batch_i+1)]),X=np.array([num_iter]),win=win,name='train',update='append')
                num_iter+=1

        val_loss=0.0
        net.eval()

        for batch_i,data in enumerate(test_loader):
            with torch.no_grad():
                images = data['image']
                key_pts = data['keypoints']
                key_pts = key_pts.view(key_pts.size(0), -1)
                images, key_pts = Variable(images), Variable(key_pts)
                if (torch.cuda.is_available()):
                    key_pts = key_pts.type(torch.cuda.FloatTensor)
                    images = images.type(torch.cuda.FloatTensor)
                    images.to(device)
                else:
                    key_pts = key_pts.type(torch.FloatTensor)
                    images = images.type(torch.FloatTensor)
                output_pts = net(images)
                loss = criterion(output_pts, key_pts)
                val_loss+=loss.item()
        val_loss/=len(test_dataset)/batch_size
        vis.line(Y=np.array([val_loss]),X=np.array([val_iter]),
                 win=win_,name='val',update='append')
        val_iter+=1
        print('loss of val is {}'.format(val_loss))


        if (epoch+1) % 50 == 0:
            torch.save(net.state_dict(), 'model_keypoints_68pts_iter_{}.pt'.format(epoch+1))
    print('Finished Training')