import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import tensorboardX as tbx

import config
from dataset import OCRDataset
from model import OCRResNet18
from loss import OCRLoss
import utils

print('PyTorch version:', torch.__version__)
print('torchvision version:', torchvision.__version__)
can_use_gpu = torch.cuda.is_available()
print('Is GPU available:', can_use_gpu)

device = torch.device('cuda' if can_use_gpu else 'cpu')

root_dir = '../../data/komonjo/200003076/'
original_image_dir = root_dir + 'images/'
resized_image_dir = root_dir + 'images_resized_320/'
log_dir = root_dir + 'logs/20190308/'

path_to_annotation_csv = root_dir + '200003076_coordinate.csv'
preprocessed_annotation_list = utils.preprocess_annotation(path_to_annotation_csv, 
                                                           original_image_dir)
utf16_to_index, index_to_utf16 = utils.make_maps_between_index_and_frequent_characters_utf16(preprocessed_annotation_list, 
                                                                                             config.N_KINDS_OF_CHARACTERS)
selected_annotation_list = utils.select_annotation_and_convert_ut16_to_index(preprocessed_annotation_list, 
                                                                             utf16_to_index)
train_annotation_list, validation_annotation_list = train_test_split(selected_annotation_list[:200],
                                                                     test_size=0.2,
                                                                     random_state=config.RANDOM_SEED)

print('The number of training data:', len(train_annotation_list))
print('The number of validation data:', len(validation_annotation_list))

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = OCRDataset(resized_image_dir, train_annotation_list, transform=tf)
validation_dataset = OCRDataset(resized_image_dir, validation_annotation_list, transform=tf)

batchsize_train = 32
batchsize_validation = batchsize_train
train_loader = DataLoader(train_dataset, batch_size=batchsize_train, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batchsize_validation)

net = OCRResNet18(5*config.N_KINDS_OF_CHARACTERS, pretrained=True)
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)
criterion = OCRLoss([1.0, 1.0, 1.0])

def train(data_loader):
    net.train()
    running_loss = 0

    running_losses = np.zeros(3)

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)

        optimizer.zero_grad()
        loss, losses = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_losses += np.array(losses)

    avarage_loss = running_loss / len(data_loader)
    average_losses = running_losses / len(data_loader)

    return avarage_loss, average_losses

def validation(data_loader):
    net.eval()
    running_loss = 0
    running_losses = np.zeros(3)

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss, losses = criterion(outputs, labels)

            running_loss += loss.item()
            running_losses += np.array(losses)

    average_loss = running_loss / len(data_loader)
    average_losses = running_losses / len(validation_loader)
    
    return average_loss, average_losses


n_epochs = 10
train_loss_list = []
train_losses_list = []
validation_loss_list = []
validation_losses_list = []
writer = tbx.SummaryWriter(log_dir + 'exp-1')

for epoch in range(n_epochs):
    train_loss, train_losses = train(train_loader)
    validation_loss, validation_losses = validation(validation_loader)
    
    # writer.add_scalar('settings/learning rate', optimizer.param_groups[0]['lr'], epoch)
    
    writer.add_scalar('train/overall loss', train_loss, epoch)
    writer.add_scalar('train/responsible loss', train_losses[0], epoch)
    writer.add_scalar('train/coordinate loss', train_losses[1], epoch)
    writer.add_scalar('train/size loss', train_losses[2], epoch)
    
    writer.add_scalar('validation/overall loss', validation_loss, epoch)
    writer.add_scalar('validation/responsible loss', validation_losses[0], epoch)
    writer.add_scalar('validation/coordinate loss', validation_losses[1], epoch)
    writer.add_scalar('validation/size loss', validation_losses[2], epoch)
    
    train_loss_list.append(train_loss)
    train_losses_list.append(train_losses)
    validation_loss_list.append(validation_loss)
    validation_losses_list.append(validation_losses)
    
    if (epoch+1) % 5 == 0:
        torch.save(net.state_dict(), log_dir + 'weight_%03d.pth' % (epoch+1))
    
    print('epoch[%3d/%3d] train_loss:%2.4f details:[resp:%1.4f coor:%1.4f size:%1.4f]'                        
         % (epoch+1, n_epochs,
            train_loss, 
            train_losses[0], train_losses[1], train_losses[2]))
    print('          validation_loss:%2.4f details:[resp:%1.4f coor:%1.4f size:%1.4f]'                                        
         % (validation_loss, 
            validation_losses[0], validation_losses[1], validation_losses[2]))

writer.close()
np.save(log_dir + 'train_loss_list.npy', np.array(train_loss_list))
np.save(log_dir + 'train_losses_list.npy', np.array(train_losses_list))
np.save(log_dir + 'validation_loss_list.npy', np.array(validation_loss_list))
np.save(log_dir + 'validation_losses_list.npy', np.array(validation_losses_list))
