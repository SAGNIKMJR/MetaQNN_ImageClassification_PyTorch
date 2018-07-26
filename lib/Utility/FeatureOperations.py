import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def spatial_pyramid_pooling(input, level):
	'''
	input: 4 channel input (bt,ch,r,c)
	level: no of levels of pooling
	returns : does spatial pyrimidal pooling and returns the output
	'''
	# TODO: implement for asymmetric/non-square feature spaces (working temporarily for square feature spaces)
	assert input.dim() == 4
	output = []
	"""
		NOTE: Sumit's implementation
	"""
	for i in range(1, level  + 1):
		kernel_size = (int(np.ceil(input.size(2) / (1.0 * i))),
						int(np.ceil(input.size(3) / (1.0 * i))))
		stride_size = (int(np.floor(input.size(2) / (1.0 * i))),
						int(np.floor(input.size(3) / (1.0 * i))))
		level_out = F.max_pool2d(input, kernel_size=kernel_size,
								stride=stride_size)
		output.append(level_out.view(input.size()[0], -1))

	# for i in range(1, level+1):
	# 	for j in range(1, i + 1):
	# 		ind1 = (int(np.floor((j-1)*input.size(2)/float(i))), int(np.ceil(j*input.size(2)/float(i))))
	# 		for k in range(1, i + 1):
	# 			ind2 = (int(np.floor((k-1)*input.size(3)/float(i))), int(np.ceil(k*input.size(3)/float(i))))
	# 			kernel_size = (ind1[1] - ind1[0], ind2[1] - ind2[0])
	# 			level_out = F.max_pool2d(input[:,:,ind1[0]:ind1[1],ind2[0]:ind2[1]], kernel_size=kernel_size)
	# 			output.append(level_out.view(input.size()[0], -1))
	final_out = torch.cat(output, 1)
	return final_out


def full_average_pooling(input):
	'''
	input: 4 channel input (bt,ch,r,c)
	returns : bt*ch*1*1 feature maps performed by averaging pooling with kernel size == input size
	'''
	assert input.dim() == 4
	return F.avg_pool2d(input, kernel_size=(input.size(2), input.size(3)))


def CenterCrop(cropTarget, cropVar):
	'''
	cropTarget: target image (the shape is deduced from this image)
	cropVar: image to be cropped
	returns : crops CropVar to the size of cropTarget by performing center crop
	'''
	cropSize = cropTarget.size()
	tw = cropSize[2]//2
	th = cropSize[3]//2
    
	varSize = cropVar.size()
	c1 = varSize[2]//2
	c2 = varSize[3]//2
    
	subW = 0
	subH = 0
    
	if cropSize[2]%2==0 and varSize[2]%2==0:
		subW = 1
        
	if cropSize[3]%2==0 and varSize[3]%2==0:
		subH = 1 
        
	cropOp = cropVar[:,:,c1-tw:c1+tw+1-subW, c2-th:c2+th+1-subH].clone()
	return cropOp


def PeriodicShuffle(x,factor):
	'''
	x: input feature map
	factor: upsampling factor
	returns : upsampled image with the mentioned factor
	'''
	btSize, ch, rows, cols = x.size()
	ch_target = ch/(factor*factor)
	ch_factor = ch/ch_target
    
    #intermediate shapes
	shape_1 = [btSize,  ch_factor // factor, ch_factor // factor, rows, cols]
	shape_2 = [btSize,  1, rows * factor, cols * factor]
    
    # reshape and transpose for periodic shuffling for each channel
	out = []
	for i in range(ch_target):
		temp = x[:, i*ch_factor:(i+1)*ch_factor, :, :]
		temp = temp.view(shape_1)
		temp = temp.permute(0,1,3,2,4)
		temp = temp.contiguous()
		temp = temp.view(shape_2)
		out.append(temp)

    # final output
	out = torch.cat(out, 1)
	return out
