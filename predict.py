# Imports here
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
import PIL
from train import check_gpu
import numpy as np

def predict(image_path, model, device, cat_to_name, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    
    model.to(device)
    model.eval();
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to(device)
    logps = model.forward(torch_image)
    ps= torch.exp(logps)
    
    top_ps, top_classes = ps.topk(topk, dim = 1)
    top_ps = np.array(top_ps.detach())[0] 
    top_classes = np.array(top_classes.detach())[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[lab] for lab in top_classes]
    top_flowers = [cat_to_name[lab] for lab in top_classes]
    
    return top_ps, top_classes, top_flowers

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    img = Image.open(image)

    # Get original dimensions
    original_width, original_height = img.size

    # Find shorter size and create settings to crop shortest side to 256
    if original_width < original_height:
        size=[256, 256**600]
    else: 
        size=[256**600, 256]
        
    img.thumbnail(size)
   

    center = original_width/4, original_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    img = img.crop((left, top, right, bottom))

    numpy_img = np.array(img)/255 

    # Normalize each color channel
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img-mean)/std
        
    # Set the color to the first channel
    numpy_img = numpy_img.transpose(2, 0, 1)
    
    return numpy_img


# function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint["arch"]](pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        
  # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    learning_rate = checkpoint['lr']
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)    
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    
    return model, optimizer

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image',type=str,help='Point to impage file for prediction.',required=True)
    parser.add_argument('--checkpoint',type=str,help='Point to checkpoint file as str.',required=True)
    parser.add_argument('--top_k',type=int, default = 1, help='Choose top K matches as int.')
    parser.add_argument('--category_names', type=str, dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', type=str, default="gpu", action="store", dest="gpu")

    args = parser.parse_args()
    
    return args
def main():
    args = arg_parser()
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    model, optimizer = load_checkpoint(args.checkpoint)
        
    device = check_gpu(gpu_arg=args.gpu);
    
    top_probs, top_labels, top_flowers = predict(args.image, model, device, cat_to_name, args.top_k)
    print(f"topK probs: {top_probs} \ntop_labels: {top_labels} \ntop_flowers : {top_flowers}")

if __name__ == '__main__':
    main()
    
