import torch
from torch import models
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import numpy as np
import json
import argparse

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.inception_v3(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.fc = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    criterion = checkpoint['criterion']
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Resize the image
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))

    # Crop the image
    left_margin = (image.width-224)/2
    bottom_margin = (image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    image = image.crop((left_margin, bottom_margin, right_margin,
                        top_margin))
    
    # Normalize the image
    np_image = np.array(image)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std

    # Move color channels to first dimension as expected by PyTorch
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def predict(image_path, model, topk=5, cat_to_name='cat_to_name.json', gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    #Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    if gpu:
        device = "cuda"
    else:
        device = "cpu"
    image = image.to(device)
    model.to(device)
    model.eval()


    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p = top_p.cpu().numpy()[0]
        top_class = top_class.cpu().numpy()[0]
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_class = [idx_to_class[each] for each in top_class]
        top_flowers = [cat_to_name[each] for each in top_class]

    return top_p, top_class, top_flowers

def main(image_path, checkpoint, topk=5, cat_to_name='cat_to_name.json', gpu=False):
    model = load_checkpoint(checkpoint)
    probs, classes, flowers = predict(image_path, model, topk, cat_to_name, gpu)
    print(probs)
    print(classes)
    print(flowers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name. That is you pass in a single image /path/to/image and return the flower name and class probability.')
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    main(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)

