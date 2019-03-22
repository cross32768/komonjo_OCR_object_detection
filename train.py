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
from loss import OCRLoss
from model import OCRResNet18, OCRResNet34, OCRResNet50, OCRVGG16, OCRVGG19
import utils


print('PyTorch version:', torch.__version__)
print('torchvision version:', torchvision.__version__)
can_use_gpu = torch.cuda.is_available()
print('Is GPU available:', can_use_gpu)


device = torch.device('cuda' if can_use_gpu else 'cpu')

log_dir = '../../data/komonjo/logs/20190322/'
selected_annotation_list, _ = utils.prepare_selected_annotation_from_dataset_indexes([6, 12])
# selected_annotation_list, _ = utils.prepare_selected_annotation_from_dataset_indexes([6, 12])
train_annotation_list, validation_annotation_list = train_test_split(selected_annotation_list,
                                                                     test_size=0.2,
                                                                     random_state=config.RANDOM_SEED)

print('The number of training data:', len(train_annotation_list))
print('The number of validation data:', len(validation_annotation_list))


tf_train = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05)], p=0.5),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
tf_validation = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = OCRDataset(train_annotation_list, transform=tf_train)
validation_dataset = OCRDataset(validation_annotation_list, transform=tf_validation)

batchsize_train = 32
batchsize_validation = batchsize_train
train_loader = DataLoader(train_dataset, batch_size=batchsize_train, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batchsize_validation)

net = OCRResNet34(5*config.N_KINDS_OF_CHARACTERS, pretrained=True)
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
criterion = OCRLoss([config.LAMBDA_RESP, config.LAMBDA_COOR, config.LAMBDA_SIZE])


def train(data_loader):
    net.train()
    running_loss = 0
    running_losses = np.zeros(3)
    
    data_loader.dataset.update_image_size(config.RESIZE_IMAGE_SIZE_CANDIDATES)

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)

        optimizer.zero_grad()
        loss, losses = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_losses += np.array(losses)

        data_loader.dataset.update_image_size(config.RESIZE_IMAGE_SIZE_CANDIDATES)

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
    average_losses = running_losses / len(data_loader)

    return average_loss, average_losses


n_epochs = 500
train_loss_list = []
train_losses_list = []
validation_loss_list = []
validation_losses_list = []
writer = tbx.SummaryWriter(log_dir + 'exp-1')

for epoch in range(n_epochs):
    train_loss, train_losses = train(train_loader)
    validation_loss, validation_losses = validation(validation_loader)

    writer.add_scalar('settings/learning rate', optimizer.param_groups[0]['lr'], epoch)

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
    
    if (epoch+1) % 200 == 0:
        optimizer.param_groups[0]['lr'] /= 10

    print('epoch[%3d/%3d] train_loss:%2.4f details:[resp:%1.4f coor:%1.4f size:%1.4f]'
          % (epoch+1, n_epochs,
             train_loss, train_losses[0], train_losses[1], train_losses[2]))
    print('          validation_loss:%2.4f details:[resp:%1.4f coor:%1.4f size:%1.4f]'
          % (validation_loss, validation_losses[0], validation_losses[1], validation_losses[2]))

writer.close()
np.save(log_dir + 'train_loss_list.npy', np.array(train_loss_list))
np.save(log_dir + 'train_losses_list.npy', np.array(train_losses_list))
np.save(log_dir + 'validation_loss_list.npy', np.array(validation_loss_list))
np.save(log_dir + 'validation_losses_list.npy', np.array(validation_losses_list))
