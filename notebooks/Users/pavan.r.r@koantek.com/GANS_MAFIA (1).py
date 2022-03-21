# Databricks notebook source
# from google.colab import drive
# drive.mount('/content/drive')

# COMMAND ----------

# from google.colab.patches import cv2_imshow

# COMMAND ----------

!git clone https://github.com/dattasiddhartha/segmented-style-transfer

# COMMAND ----------

!git clone https://github.com/levindabhi/cloth-segmentation.git

# COMMAND ----------

# MAGIC %sh git clone https://github.com/levindabhi/cloth-segmentation.git --depth 1 --branch=master /dbfs/FileStore/TheData2/

# COMMAND ----------

# MAGIC %sh git clone https://github.com/dattasiddhartha/segmented-style-transfer --depth 1 --branch=master /dbfs/FileStore/TheData/

# COMMAND ----------

import sys
sys.path.append("/Workspace/Repos/roopesh.mangeshkar@koantek.com/Revamp")

# COMMAND ----------

sys.path.append("/Workspace/Repos/roopesh.mangeshkar@koantek.com/Revamp3")

# COMMAND ----------

import sys
sys.path.append("/Workspace/Repos/roopesh.mangeshkar@koantek.com/cloth-segmentation")

# COMMAND ----------

sys.path.append("/Workspace/Repos/roopesh.mangeshkar@koantek.com/segmented-style-transfer")

# COMMAND ----------

import os
# from tqdm import tqdm
from tqdm.notebook import tqdm
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from networks import U2NET


#from PIL import Image
import matplotlib.pyplot as plt
import matplotlib, random
import torch, torchvision
import torchvision.transforms as T
import numpy as np
import numpy.ma as ma
import cv2
from vision.faststyletransfer_eval import FasterStyleTransfer
import collections
import ntpath

# COMMAND ----------

# MAGIC %scala
# MAGIC     dbutils.fs.mount(
# MAGIC       source = "wasbs://revamp-dataset-1@koanteklndblob.blob.core.windows.net/",
# MAGIC       mountPoint = "/mnt/revamp-dataset-1",
# MAGIC extraConfigs=Map("fs.azure.account.key.koanteklndblob.blob.core.windows.net" -> "HO+CR/uOaacXUTm5P/A0cOPjaEaCsSV05cNZhL5U67OhflS2884VVzMCBLsETTprNQmI4Rr58qVn+AStCIbG6g=="))

# COMMAND ----------

dbutils.fs.ls()

# COMMAND ----------

# x=cv2.imread('/content/drive/MyDrive/sample_tendulkar.jpg')

# COMMAND ----------

device='cuda'

# COMMAND ----------

image_to_mask_mapping={}

# COMMAND ----------

image_dir = '/dbfs/mnt/revamp-dataset-1/Input_Folder'
mask_dir = '/dbfs/mnt/revamp-dataset-1/Mask_Folder'
output_dir='/dbfs/mnt/revamp-dataset-1/Output_Folder'
style_transfer_mask_dir='/dbfs/mnt/revamp-dataset-1/Styled_Folder'
checkpoint_path = '/dbfs/mnt/revamp-dataset-1/cloth_segm_u2net_latest.pth'

# COMMAND ----------

transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)


# COMMAND ----------

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
# net = net.to(device)
net = net.eval()

# COMMAND ----------

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

# COMMAND ----------

palette = get_palette(4)

# COMMAND ----------

for i,j in image_to_mask_mapping.items():
  plt.imshow(cv2.imread(i))
  plt.show()
  plt.imshow(cv2.imread(j))
  plt.show()
  

# COMMAND ----------

# x=cv2.imread('/content/drive/MyDrive/lufi_sample.jpg')

# COMMAND ----------

# print(x)

# COMMAND ----------

""" img = Image.open('/content/drive/MyDrive/lufi_sample.jpg').convert('RGB')
img_size = img.size
img = img.resize((768, 768), Image.BICUBIC)
image_tensor = transform_rgb(img)
image_tensor = torch.unsqueeze(image_tensor, 0)

output_tensor = net(image_tensor.to(device))
output_tensor = F.log_softmax(output_tensor[0], dim=1)
output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
output_tensor = torch.squeeze(output_tensor, dim=0)
output_tensor = torch.squeeze(output_tensor, dim=0)
output_arr = output_tensor.cpu().numpy()

output_img = Image.fromarray(output_arr.astype('uint8'), mode='L')
output_img = output_img.resize(img_size, Image.BICUBIC)

output_img.putpalette(palette) """


# COMMAND ----------

images_list = sorted(os.listdir(image_dir))
pbar = tqdm(total=len(images_list))
i=0
for image_name in images_list:
    i+=1
    img = Image.open(os.path.join(image_dir, image_name)).convert('RGB')
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    
    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    output_img = Image.fromarray(output_arr.astype('uint8'), mode='L')
    output_img = output_img.resize(img_size, Image.BICUBIC)
    
    output_img.putpalette(palette)

    # d[os.path.join(image_dir, image_name)]=output_img
    output_img.save(os.path.join(mask_dir, image_name[:-4]+'_generated.png'))
    image_to_mask_mapping
    pbar.update(1)
    image_to_mask_mapping[os.path.join(image_dir, image_name)]=os.path.join(mask_dir, image_name[:-4]+'_generated.png')
    if i==100:
      break
pbar.close()

# COMMAND ----------

for i,j in image_to_mask_mapping.items():
  cv2_imshow(cv2.imread(i))
  cv2_imshow(cv2.imread(j))

# COMMAND ----------

model = torch.load('/content/drive/MyDrive/GANS/Models/cloth_segm_u2net_latest.pth')

# COMMAND ----------

import cv2
from matplotlib import pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overlap Image and Mask

# COMMAND ----------

for a,b in image_to_mask_mapping.items():
  masks = cv2.imread(b)
  img = cv2.imread(a)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # for i in range(len(masks)):
    # rgb_mask = random_colour_masks(masks[i])
  img = cv2.addWeighted(img, 0.4, masks, 1, 0)
    #cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # no bounding boxes required
    # cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
  plt.figure(figsize=(20,30))
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()
  
  # return img

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mapping masks to input

# COMMAND ----------

for i,j in image_to_mask_mapping.items():
    # print(i,j)
    original_image=cv2.imread(i)
    img_original_rbg = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_mask=cv2.imread(j)
    break


# COMMAND ----------

def mapping(img_path='./payload/IMG-20200401-WA0002.jpg'):
  img1=cv2.imread(img_path)
  mask_1=cv2.imread(image_to_mask_mapping[img_path])
  return img1,mask_1

# COMMAND ----------

x,y=mapping('/content/drive/MyDrive/GANS/Input_Folder/modi.jpg')

# COMMAND ----------

# image_to_mask_mapping['/content/drive/MyDrive/new dataset/dataset/test/00a8764cff12b2e849c850f4be5608bc.jpg']

# COMMAND ----------

def mask_segments(img_path='./payload/IMG-20200401-WA0002.jpg'):
    img_original,masks = mapping(img_path)
    img_original_rbg = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    img = transform(img_original)
    img_rgb = transform(img_original_rbg)
    # pred = model([img])
    # print(pred[0])
    # print("Finished image segmentation") 
    # masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    masks=cv2.cvtColor(masks, cv2.COLOR_BGR2GRAY)
    # print("Returned segments: ", len(masks))
    
    return img_original_rbg, img_rgb, masks

# COMMAND ----------

# cv2_imshow(cv2.cvtColor(original_mask, cv2.COLOR_BGR2GRAY))

# COMMAND ----------

# plt.imshow(original_image)
# plt.show()

# COMMAND ----------

def PartialStyleTransfer(segment = 0, img_path='./payload/IMG-20200401-WA0002.jpg', style_path="./fast_neural_style_transfer/models/mosaic_style__200_iter__vgg19_weights.pth"):

    print("Started partial style transfer")

    # mode can be 'styled' or 'color'

    # return indices on number of segments
    img_original_rbg, img_rgb, masks = mask_segments(img_path)
    plt.imshow(img_original_rbg)
    plt.show()
    print(len(masks))
    if len(masks) > 0:
        mask = masks
        print(mask.shape)
        # print mask of image with the original image pixels
        img_array = np.array(img_original_rbg[:,:,:])
        img_array_floating = np.array(img_rgb[:,:,:])
        # if False, set as 0 (black)

        masked_img = []
        for h in range(img_original_rbg.shape[0]):
            sub_masked_img = []
            for i in range(img_original_rbg.shape[1]):
                tmp=[]
                for j in range(img_original_rbg.shape[2]):
                    if mask[h][i] == False:
                        tmp.append(float(0))
                    else:
                        tmp.append(img_array_floating[j][h][i])
                sub_masked_img.append(tmp)
            masked_img.append(sub_masked_img)      

        masked_img_array = np.array(masked_img)
        plt.imshow(masked_img_array[:,:,:]) # Export this mask image for style transfer
        plt.show()

        matplotlib.image.imsave(str(mask_dir + "/" + ntpath.basename(img_path)[:-4]+str("_MASK")+".png"), masked_img_array)

        FasterStyleTransfer(style_path, str(mask_dir + "/" + ntpath.basename(img_path)[:-4]+str("_MASK")+".png"), str(style_transfer_mask_dir + "/" + ntpath.basename(img_path)[:-4]+str("_FST")+".png"))

        style_img = Image.open(str(style_transfer_mask_dir + "/" + ntpath.basename(img_path)[:-4]+str("_FST")+".png"))
        plt.imshow(style_img)
        plt.show()

    return style_img, img_array_floating, img_array

# COMMAND ----------

def PixelRemoved(img_path='./payload/IMG-20200401-WA0002.jpg'):
    transform = T.Compose([T.ToTensor()])
    img_original_rbg = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_rgb = transform(img_original_rbg)
    img_array_floating = np.array(img_rgb[:,:,:])
    style_img_original = Image.open(str(style_transfer_mask_dir + "/" + ntpath.basename(img_path)[:-4]+str("_FST")+".png"))
    WIDTH, HEIGHT = cv2.cvtColor(cv2.imread(str(mask_dir + "/" + ntpath.basename(img_path)[:-4]+str("_MASK")+".png")), cv2.COLOR_BGR2RGB).shape[1], cv2.cvtColor(cv2.imread(str(mask_dir + "/" + ntpath.basename(img_path)[:-4]+str("_MASK")+".png")), cv2.COLOR_BGR2RGB).shape[0]
    style_img_rbg = cv2.resize(cv2.cvtColor(cv2.imread(str(style_transfer_mask_dir + "/" + ntpath.basename(img_path)[:-4]+str("_FST")+".png")), cv2.COLOR_BGR2RGB), (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC) # FST reshaped the dimension, this lines reshapes back to consistent dimensions
    styled_img = transform(style_img_original)
    styled_img_rgb = transform(style_img_rbg)
    # remove most frequent pixel
    pix_remove = list(dict(collections.Counter(np.hstack(np.hstack(styled_img_rgb))).most_common()).keys())[0]
    # img_array = np.array(img_original_rbg[:,:,:])
    styled_img_rgb_floating = np.array(styled_img_rgb[:,:,:])
    masked_img = []
    # When it is detected to be a background pixed, a background pixel from original image is inserted
    for h in range(style_img_rbg.shape[0]):
        sub_masked_img = []
        for i in range(style_img_rbg.shape[1]):
            tmp=[]
            for j in range(style_img_rbg.shape[2]):
                if (float(styled_img_rgb[j][h][i]) > float(pix_remove)-0.1) and (float(styled_img_rgb[j][h][i]) < float(pix_remove)+0.1):
                    tmp.append(img_array_floating[j][h][i])
                else:
                    tmp.append(styled_img_rgb_floating[j][h][i])
            sub_masked_img.append(tmp)
        masked_img.append(sub_masked_img) 
    masked_img_array = np.array(masked_img)
    plt.imshow(masked_img_array[:,:,:])
    matplotlib.image.imsave(str(output_dir + "/" + ntpath.basename(img_path)[:-4]+str("_MASK_FST")+".png"), masked_img_array)
    return masked_img_array

# COMMAND ----------

cd /content/segmented-style-transfer

# COMMAND ----------

class GansMafia:
  device='cuda'
  image_dir = '/content/drive/MyDrive/GANS/Input_Folder'
  mask_dir = '/content/drive/MyDrive/GANS/Mask_Folder'
  output_dir='/content/drive/MyDrive/GANS/Output_Folder'
  style_transfer_mask_dir='/content/drive/MyDrive/GANS/Styled_Folder'
  checkpoint_path = '/content/drive/MyDrive/GANS/Models/cloth_segm_u2net_latest.pth'

  def __init__(self, inp_img):
    self.inp_img = inp_img
    self.img_name = ntpath.basename(self.inp_img)[:-4]
    
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    
    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint_mgpu(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()
    
  def get_palette(self, num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette
  
  def mask_dress(self):
    img = Image.open(self.inp_img).convert('RGB')
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    
    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    output_img = Image.fromarray(output_arr.astype('uint8'), mode='L')
    output_img = output_img.resize(img_size, Image.BICUBIC)
    
    output_img.putpalette(palette)

    # d[os.path.join(image_dir, image_name)]=output_img
    output_img.save(os.path.join(mask_dir, image_name[:-4]+'_generated.png'))
    image_to_mask_mapping
    #pbar.update(1)
    image_to_mask_mapping[os.path.join(image_dir, image_name)]=os.path.join(mask_dir, image_name[:-4]+'_generated.png')
  
  def getInpImg(self):
    return self.inp_img

# COMMAND ----------

GansMafia('/a/b/c').getInpImg()

# COMMAND ----------

style_img, img_array_floating, img_array = PartialStyleTransfer(segment = 13, img_path='/content/drive/MyDrive/GANS/Input_Folder/john.jpg', style_path="./vision/fast_neural_style_transfer/models/mosaic.pth")
masked_img_array = PixelRemoved(img_path='/content/drive/MyDrive/GANS/Input_Folder/john.jpg')

# COMMAND ----------

!pwd

# COMMAND ----------

  image_to_mask_mapping.items()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

def PixelRemoved(img_path='./payload/IMG-20200401-WA0002.jpg'):

    transform = T.Compose([T.ToTensor()])
    
    # img_original_rbg = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_original_rbg=cv2.imread(img_path)
    img_rgb = transform(img_original_rbg)
    img_array_floating = np.array(img_rgb[:,:,:])
    
    style_img_original = Image.open(str(img_path[:-4]+str("_FST")+".png"))
    WIDTH, HEIGHT = cv2.cvtColor(cv2.imread(str(img_path[:-4]+str("_MASK")+".png")), cv2.COLOR_BGR2RGB).shape[1], cv2.cvtColor(cv2.imread(str(img_path[:-4]+str("_MASK")+".png")), cv2.COLOR_BGR2RGB).shape[0]
    style_img_rbg = cv2.resize(cv2.cvtColor(cv2.imread(str(img_path[:-4]+str("_FST")+".png")), cv2.COLOR_BGR2RGB), (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC) # FST reshaped the dimension, this lines reshapes back to consistent dimensions
    styled_img = transform(style_img_original)
    styled_img_rgb = transform(style_img_rbg)

    # remove most frequent pixel
    pix_remove = list(dict(collections.Counter(np.hstack(np.hstack(styled_img_rgb))).most_common()).keys())[0]

    # img_array = np.array(img_original_rbg[:,:,:])
    styled_img_rgb_floating = np.array(styled_img_rgb[:,:,:])

    masked_img = []
    # When it is detected to be a background pixed, a background pixel from original image is inserted
    for h in range(style_img_rbg.shape[0]):
        sub_masked_img = []
        for i in range(style_img_rbg.shape[1]):
            tmp=[]
            for j in range(style_img_rbg.shape[2]):
                if (float(styled_img_rgb[j][h][i]) > float(pix_remove)-0.1) and (float(styled_img_rgb[j][h][i]) < float(pix_remove)+0.1):
                    tmp.append(img_array_floating[j][h][i])
                else:
                    tmp.append(styled_img_rgb_floating[j][h][i])
            sub_masked_img.append(tmp)
        masked_img.append(sub_masked_img) 

    masked_img_array = np.array(masked_img)
    print("this is the wrong image")
    plt.imshow(cv2.cvtColor(masked_img_array[:,:,:], cv2.COLOR_BGR2RGB))
    # plt.imshow(masked_img_array[:,:,:])

    matplotlib.image.imsave(str(img_path[:-4]+str("_MASK+FST")+".png"), masked_img_array)
    
    return masked_img_array

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

cv2_imshow(original_mask[original_mask>1])

# COMMAND ----------

cv2_imshow(masks)

# COMMAND ----------

img_array = np.array(original_image[:,:,:])
img_array_floating = np.array(original_image[:,:,:])

masked_img = []
for h in range(original_image.shape[0]):
    sub_masked_img = []
    for i in range(original_image.shape[1]):
        tmp=[]
        for j in range(original_image.shape[2]):
          # print(original_mask[h][i])
          if original_mask[h][i].all() == False:
                tmp.append(float(0))
          else:
                tmp.append(img_array_floating[i][h][j])
        sub_masked_img.append(tmp)
    masked_img.append(sub_masked_img) 
masked_img_array = np.array(masked_img)
plt.imshow(masked_img_array[:,:,:]) # Export this mask image for style transfer
plt.show()

# COMMAND ----------

img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# COMMAND ----------

# print mask of image with the original image pixels
img_array = np.array(original_image[:,:,:])
img_array_floating = np.array(img_rgb[:,:,:])
# if False, set as 0 (black)

masked_img = []
for h in range(img_original_rbg.shape[0]):
    sub_masked_img = []
    for i in range(img_original_rbg.shape[1]):
        tmp=[]
        for j in range(img_original_rbg.shape[2]):
            # print(original_mask[h][i])
            if original_mask[h][i].any() == False:
                tmp.append(float(0))
            else:
                tmp.append(img_array_floating[j][h][i])
        sub_masked_img.append(tmp)
    masked_img.append(sub_masked_img)      

masked_img_array = np.array(masked_img)
plt.imshow(masked_img_array[:,:,:]) # Export this mask image for style transfer
plt.show()


# COMMAND ----------

original_mask

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

for a,b in image_to_mask_mapping.items():
  masks = cv2.imread(b)
  # plt.figure(figsize=(20,30))
  # plt.imshow(masks)
  pix_remove = list(dict(collections.Counter(np.hstack(np.hstack(masks))).most_common()).keys())[0]
  print(pix_remove)
  break

# COMMAND ----------



# COMMAND ----------

style_img, img_array_floating, img_array = PartialStyleTransfer(segment = 13, img_path='/content/drive/MyDrive/new dataset/dataset/test/003d41dd20f271d27219fe7ee6de727d.jpg', style_path="./vision/fast_neural_style_transfer/models/mosaic.pth")

# COMMAND ----------

