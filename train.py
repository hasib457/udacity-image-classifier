# Imports here

import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

# check availability of GPU
def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

# Transforming of train data
def train_transformer(train_dir):
   train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
   train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
   return train_data

# Transforming of test data
def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
    
# Data loader
def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader

#  Pimary moodel loader and specify model architecture
def primaryloader_model(architecture="vgg11"):
    
    model = models.__dict__[architecture](pretrained=True)
    inputs_feat = model.classifier[0].in_features   
    model.arch = architecture
    
    for param in model.parameters():
        param.requires_grad = False 
        
    return model, inputs_feat

#  Initialize model and create a new model classifier
def initiat_model(model, inputs_feat):
    # build a new classifier

    model.classifier = nn.Sequential(OrderedDict([
                        ("inputs", nn.Linear(inputs_feat, 1200)),
                        ("relu1", nn.ReLU()),
                        ("dropout", nn.Dropout(0.3)),
                        ("hidden", nn.Linear(1200, 500)),
                        ("relu2", nn.ReLU()),
                        ("hidden2", nn.Linear(500, 102)),
                        ('output', nn.LogSoftmax(dim=1))# For using NLLLoss()
                        ]))
    
    return model

# Train model
def train(model, trainloader, validloader, device, criterion, optimizer, epochs):
    running_loss = 0
    step = 0

    for epoch in range(epochs):

        for images, labels in trainloader:
            step +=1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model.forward(images)
            loss = criterion(logits,  labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                valid_loss = 0
                accuracy = 0
                model.eval()

                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model.forward(images)
                    batch_loss = criterion(logits,  labels)
                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logits)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            if step == 10:        
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss:.3f}.. "
                      f"validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"validation accuracy: {accuracy/len(validloader):.3f}")
                step = 0
                
            running_loss = 0

            model.train()
    return model  

#  validate model
def test(model, testloader, device):
    with torch.no_grad():
        model.eval()
        correct, total = 0,0
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logits = model.forward(images)

            # Calculate accuracy
            ps = torch.exp(logits)
            top_p, top_class = ps.topk(1, dim=1)
            correct += (top_class == labels.view(*top_class.shape)).sum().item()
            total += labels.size(0)
            
    return correct/total


# Save the checkpoint 
def save_checkpoint(model, optimizer, lr, train_set, checkpoint_dir ):
    model.class_to_idx = train_set.class_to_idx
    torch.save({ "arch": model.arch,
                 "lr": lr,
                 "classifier": model.classifier,
                 'class_to_idx':model.class_to_idx,
                 'state_dict':model.state_dict(),
                 'optimizer_dict': optimizer.state_dict()
                 }, checkpoint_dir)

    
# arg parser for user options 
def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg11", type = str)
    parser.add_argument('--data_dir', dest="data_dir", action="store", type = str, default="flowers")
    parser.add_argument('--save_dir', dest="save_dir", action="store", type = str, default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", type= float, default=0.001)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", type = str, default="gpu")
    args = parser.parse_args()
    return args


# main function 
def main():
    args = arg_parser()
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = test_transformer(train_dir)
    valid_data = train_transformer(valid_dir)
    test_data = train_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    # specify arch     
    model, inputs_feat = primaryloader_model(architecture=args.arch)
    
    #build new classifier
    model = initiat_model(model, inputs_feat)
    
    # choose device
    device = check_gpu(args.gpu)
    model.to(device);
    
    #train model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= args.learning_rate)
    
    model = train(model, trainloader, validloader, device, criterion, optimizer, epochs = args.epochs)
    
    #test model
    acc = test(model, testloader, device)
    print(f"model accuracy on test data : {acc:.3f}")
    
    #save checkpoint
    save_checkpoint(model, optimizer, args.learning_rate, train_data,  checkpoint_dir = args.save_dir )

if __name__ == '__main__':
    main()
