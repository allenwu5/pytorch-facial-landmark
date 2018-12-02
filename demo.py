import torch
import torch.nn as nn
import torch.optim as optim
from models import *
from model_train import net_sample_output, train_net
from model_evaluate import visualize_output

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

net = ResNet18(136).to(device)
net.load_state_dict(torch.load('model_keypoints_68pts_iter_500.pt'))
net.eval()

# get a sample of test data
with torch.no_grad():
    test_images, test_outputs, gt_pts = net_sample_output(net, device)
    visualize_output(test_images, test_outputs,gt_pts)

