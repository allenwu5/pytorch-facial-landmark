import torch
import torch.nn as nn
import torch.optim as optim
from models import *
from model_train import net_sample_output, train_net
from model_evaluate import visualize_output

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

net = ResNet18(136).to(device)
#net.load_state_dict(torch.load('saved_models/my_keypoints_model.pt'))
# define loss and optimisation functions
criterion = nn.MSELoss().cuda()
#criterion=nn.SmoothL1Loss()
initial_lr=0.0001
optimizer = optim.Adam(net.parameters(), lr=initial_lr, amsgrad=True, weight_decay=0)

# train the network
n_epochs = 500
train_net(net, device, criterion, optimizer, n_epochs)

# get a sample of test data
test_images, test_outputs, gt_pts = net_sample_output(net, device)

visualize_output(test_images, test_outputs, gt_pts)

# after training, save the model parameters in the dir 'saved_models'
torch.save(net.state_dict(), 'model_keypoints_68pts_iter_final.pt')
