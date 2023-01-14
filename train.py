from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

def main(data_dir,save_dir,arch,learning_rate,epochs,gpu):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],),
    ])

    # Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)

    # Load the pre-trained model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        classifier = nn.Sequential(
                                nn.Linear(25088, 4096),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(4096, 1024),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(1024, 102),
                                nn.LogSoftmax(dim=1)
        )
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        classifier = nn.Sequential(
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 102),
                                nn.LogSoftmax(dim=1)
        )

    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        classifier = nn.Sequential(
                                nn.Linear(9216, 4096),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(4096, 1024),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(1024, 102),
                                nn.LogSoftmax(dim=1)
        )

    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        classifier = nn.Sequential(
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(128, 102),
                                nn.LogSoftmax(dim=1)
        )

    else:
        model = models.inception_v3(pretrained=True)
        classifier = nn.Sequential(
                                nn.Linear(2048, 1024),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 102),
                                nn.LogSoftmax(dim=1)
                        )


    # Set the model in evaluation mode
    model.eval()

    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replacing the model's classifier with the new classifier
    model.fc = classifier

    # set the classifier's parameters to be trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if learning_rate == None:
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    else:
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)



    # Define the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # set the number of training epochs
    if epochs == None:
        num_epochs = 150
    else:
        num_epochs = epochs


    # move the model to the device (cuda or cpu)
    if gpu == True:
        model.to('cuda')
        device = 'cuda'
    else:
        model.to('cpu')
        device = 'cpu'

    # loop over the number of epochs
    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        accuracy = 0
        
        # loop over the training data
        for inputs, labels in dataloaders:
            # move the input and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            # backward pass
            loss.backward()
            
            # optimization step
            optimizer.step()
            
            # update the training loss
            train_loss += loss.item()
            
        # loop over the validation data
        for inputs, labels in validloaders:
            # move the input and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # forward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            # update the validation loss
            valid_loss += loss.item()
            
            # calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        # calculate average loss and accuracy
        train_loss = train_loss / len(dataloaders)
        valid_loss = valid_loss / len(validloaders)
        accuracy = accuracy / len(validloaders)

        scheduler.step(valid_loss)
        
        # print out the statistics
        print(f'Epoch {epoch+1}/{num_epochs}.. '
            f'Train loss: {train_loss:.3f}.. '
            f'Valid loss: {valid_loss:.3f}.. '
            f'Valid accuracy: {accuracy:.3f}')


    #Saving the checkpoint 
    model.class_to_idx = image_datasets.class_to_idx
    checkpoint = {'input_size': 2048,
                    'output_size': 102,
                    'hidden_layers': [1024, 512, 256, 128],
                    'state_dict': model.state_dict(),
                    'class_to_idx': model.class_to_idx,
                    'classifier': model.fc,
                    'epochs': num_epochs,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'criterion': criterion
                    }
    if save_dir:
        path = save_dir + '/checkpoint.pth'
    else:
        path = 'checkpoint.pth'
    torch.save(checkpoint, path)

    print('Model saved to checkpoint.pth')

def get_input_args():
    """
    Retrieves and parses the 6 command line arguments provided by the user when they run the program from a terminal window. This function returns these arguments as an ArgumentParser object.
    Command Line Arguments:
      1. Directory path to a folder of images (str)
      2. CNN Model Architecture (str)
      3. Learning Rate (float)
      4. Number of training epochs (int)
      5. Use GPU for training (bool)
      6. Directory path to save the checkpoint (str)
    This function uses Python's argparse module to create and define these command line arguments. If the user fails to provide some or all of the 6 arguments, then the default values are used for the missing arguments. These defaults are defined within the function as constants. 
    """
    parser = argparse.ArgumentParser()


    parser.add_argument('data_dir', type = str, help = 'path to the folder of images')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'CNN Model Architecture')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate')
    parser.add_argument('--epochs', type = int, default = 150, help = 'Number of training epochs')
    parser.add_argument('--gpu', type = bool, default = True, help = 'Use GPU for training')
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Directory path to save the checkpoint')
    return parser.parse_args()

if __name__ == '__main__':
    # get the arguments for all the varialbes in the main function
    args = get_input_args()
    # call the main function
    main(args.data_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu, args.save_dir)