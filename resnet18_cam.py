#!/usr/bin/env python
# coding: utf-8

# In[19]:


import torch
import torchvision.transforms as transforms

from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn, optim
import torch.nn. functional as F
import os

import numpy as np
import matplotlib.pyplot as plt

is_pretrained=False
model_name = 'resnet18'
# 수정 : inception 모델 갖고오기 
model = models.resnet18(pretrained=is_pretrained)
model.fc.out_features=7
print(model)


# In[32]:


dataset_names=['pacs']
domains = {
            'pacs' : ['art_painting', 'cartoon', 'photo', 'sketch']
          }

source_target_domains = {}
#수정 : inception 모델에 맞게 layer 설정 
#target_layers=[2,7,14,21,28]
# cam_models=['GradCAM', 'GradCAMpp','SmoothGradCAMpp','ScoreCAM']

n_target_layers=1
#cam_models=['ScoreCAM']#'GradCAM', 'GradCAMpp','SmoothGradCAMpp'] 
cam_models=['CAM']#'GradCAM', 'GradCAMpp','SmoothGradCAMpp'] 


for idx1, dataset_name in enumerate(dataset_names):
    
    dataset_soruce_target_domains=[]
    
    for j in range(len(domains[dataset_name])):
        source_domain=''
        target_domain=''
        for idx,domain in enumerate(domains[dataset_name]):
            if idx!=j:
                source_domain+=domain+'+'
            else:
                target_domain=domain
        dataset_soruce_target_domains.append(source_domain[:-1]+'('+target_domain+')')
    source_target_domains[dataset_name]=dataset_soruce_target_domains
print(source_target_domains)
# source_target_domains['pacs']=[source_target_domains['pacs'][-1]]
# print(source_target_domains)


# In[5]:


get_ipython().system('pip install torchsummary')
import torchsummary

torchsummary.summary(model, (3,224,224), device='cpu')


# In[33]:


import skimage.transform

from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torchvision.utils import save_image

from cam import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM
from utils.visualize import visualize, reverse_normalize
from utils.imagenet_labels import label2idx, idx2label

import glob
import sys


# In[34]:


tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
       mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )])

preprocess = tf


# In[35]:


#각 dataset의 class_index, class_name 정보 얻어오기 
class_name_index_dict={}
for dataset_name in dataset_names:
    dataset_tmp=ImageFolder(root=os.path.join(dataset_name,'sketch'), transform = tf)
    new_class_name_index={}
    for item in dataset_tmp.class_to_idx.items():
        new_class_name_index[item[1]]=item[0]
    class_name_index_dict[dataset_name]= new_class_name_index
    
print(class_name_index_dict)

layers = ['model.layer4']


# In[36]:


#class_index를 받아서 class_name을 반환하는 함수 
def convertLabelIdxToName(prediction_idx, dataset_name):
    return class_name_index_dict[dataset_name][prediction_idx]


# In[31]:



#is_pretrained=True
#model = models.vgg16(pretrained=is_pretrained)
#model.classifier[6].out_features=7

for dataset_name in dataset_names: # 데이터셋종류 만큼 반복 
    for domain in source_target_domains[dataset_name]: # 각 domain이 한번씩 target_doamin이 되도록 반복 
        
              
        #수정 : domain 이름 수정 
        #수정 : 경로 수정하기 
        PATH = '{}/'.format(model_name)+dataset_name+'/'+domain+'/'
        MODEL_PATH=PATH+'transfer_{}.pth'.format(model_name)
        
        #test_image 들 불러오기 (legacy 폴더의 파일은 빼고)
        test_image_list=glob.glob(PATH+'cam/input/*')
        
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print(MODEL_PATH)
        
        for target_layer_idx in range(n_target_layers):#target_layer만큼 반복 
            
            for test_image in test_image_list:#test_image만큼 반복 
                
                image=Image.open(test_image)                    
                test_image_name=image.filename.split('/')[-1].split('.')[0]               
                
                for cam_model in cam_models: #cam_model만큼 반복 
                    
                    #cam model에 tensor변수가 들어갈 때마다 tensor가 변형되서 여기에 있어야함 
                    # convert image to tensor
                    tensor = preprocess(image)
                    # reshape 4D tensor (N, C, H, W)
                    tensor = tensor.unsqueeze(0)
                    # the target layer you want to visualize
                    if target_layer_idx==0:
                        target_layer = model.layer4
                    elif target_layer_idx==1:
                        target_layer = model.layer3
                    
                    if cam_model=='CAM':
                        wrapped_model=CAM(model, target_layer)
                    elif cam_model=='GradCAM':
                        wrapped_model=GradCAM(model, target_layer)
                    elif cam_model=='GradCAMpp':
                        wrapped_model=GradCAMpp(model, target_layer)
                    elif cam_model=='SmoothGradCAMpp':
                        #n_sample: the number of samples
                        #stdev_spread: standard deviationß
                        wrapped_model=SmoothGradCAMpp(model, target_layer,n_samples=1,stdev_spread=0.15)
                    elif cam_model=='ScoreCAM':
                        wrapped_model = ScoreCAM(model, target_layer)
                    
                    HEATMAP_IMAGE_SAVE_PATH=PATH+'cam/output/'+cam_model+'/'
                    #cam과 모델이 prediction한 class index return
                    cam, idx = wrapped_model(tensor)    
                    
                    # visualize only cam
                    #imshow(cam.squeeze().numpy(), alpha=0.5, cmap='jet')

                    # reverse normalization for display
                    img = reverse_normalize(tensor)

                    heatmap = visualize(img, cam)

                    try:
                        if not os.path.exists(HEATMAP_IMAGE_SAVE_PATH):
                            os.makedirs(HEATMAP_IMAGE_SAVE_PATH)
                    except:
                        print('Error : Creating directory. '+ HEATMAP_IMAGE_SAVE_PATH)
                        
                    #test_image_label=''
                    test_image_prediction=convertLabelIdxToName(idx,dataset_name) # test_image에 대한 prediction 
                    #print('Test image prediction: +test_image_prediction)

                    # save image
                    save_image(heatmap, HEATMAP_IMAGE_SAVE_PATH+'{}.png'.format(dataset_name+'_'+domain+'_'+test_image_name+'_'+layers[target_layer_idx]+'_'+'predict('+test_image_prediction+')'+'_'+cam_model))
                    print('Current info: ',dataset_name,domain,target_layer_idx, cam_model)
                    print("="*50)
                    
                           
            



# In[ ]:


# 뽑을거
# 2, 7, 14, 21, 28

