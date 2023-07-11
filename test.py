import os
import numpy as np
import torch
from os import listdir
import torch.nn as nn
from torch.autograd import Variable
from option import  opt
from data_utils import is_image_file
from crossformer11 import CrossFormer
import scipy.io as scio  
from eval import PSNR, SSIM, SAM
import time  
                   
def main():

    input_path = 'D:/study/数据集/CAVE处理后/×2/test/'
    out_path = './result/'
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)      
    PSNRs = []
    SSIMs = []
    SAMs = []


    if not os.path.exists(out_path):
        os.makedirs(out_path)
                    
    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    model = CrossFormer(opt)

    if opt.cuda:
        model = nn.DataParallel(model).cuda()
        
    checkpoint  = torch.load(opt.model_name)
    model.load_state_dict(checkpoint['model']) 
    model.eval()
    images_name = [x for x in listdir(input_path) if is_image_file(x)]           
    T = 0        
    for index in range(len(images_name)):
        with torch.no_grad():
            mat = scio.loadmat(input_path + images_name[index])
            hyperLR = mat['LR'].astype(np.float32).transpose(2,0,1)
            HR = mat['HR'].astype(np.float32).transpose(2,0,1)

            input = Variable(torch.from_numpy(hyperLR).float()).contiguous().view(1, -1, hyperLR.shape[1], hyperLR.shape[2])
            SR = np.array(HR).astype(np.float32)

            if opt.cuda:
               input = input.cuda()

            localFeats = []
            start = time.time()
            for i in range(11):
                if i == 0:
                    x = input[:, 0:5, :, :]
                    # y = input[:,0,:,:]
                    # new_label = label[:, 0:2, :, :]
                elif i == 1:
                    x = input[:, 1:6, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 2:5, :, :]
                elif i == 2:
                    x = input[:, 4:9, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 5:8, :, :]
                elif i == 3:
                    x = input[:, 7:12, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 8:11, :, :]
                elif i == 4:
                    x = input[:, 10:15, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 11:14, :, :]
                elif i == 5:
                    x = input[:, 13:18, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 14:17, :, :]
                elif i == 6:
                    x = input[:, 16:21, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 17:20, :, :]
                elif i == 7:
                    x = input[:, 19:24, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 20:23, :, :]
                elif i == 8:
                    x = input[:, 22:27, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 23:26, :, :]

                elif i == 9:
                    x = input[:, 25:30, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 26:29, :, :]
                elif i == 10:
                    x = input[:, 26:31, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 29:31, :, :]
                output, localFeats = model(x, localFeats, i)
                if i == 0:
                    SR[0:2, :, :] = output.cpu().data[0].numpy()
                elif i == 1:
                    SR[2:5, :, :] = output.cpu().data[0].numpy()
                elif i == 2:
                    SR[5:8, :, :] = output.cpu().data[0].numpy()
                elif i == 3:
                    SR[8:11, :, :] = output.cpu().data[0].numpy()
                elif i == 4:
                    SR[11:14, :, :] = output.cpu().data[0].numpy()
                elif i == 5:
                    SR[14:17, :, :] = output.cpu().data[0].numpy()
                elif i == 6:
                    SR[17:20, :, :] = output.cpu().data[0].numpy()
                elif i == 7:
                    SR[20:23, :, :] = output.cpu().data[0].numpy()
                elif i == 8:
                    SR[23:26, :, :] = output.cpu().data[0].numpy()

                elif i == 9:
                    SR[26:29, :, :] = output.cpu().data[0].numpy()
                elif i == 10:
                    SR[29:31, :, :] = output.cpu().data[0].numpy()
        end = time.time()   
        print (end-start)
        T = T + (end - start)
                   
        SR[SR<0] = 0             
        SR[SR>1.] = 1.
        psnr = PSNR(SR, HR)
        ssim = SSIM(SR, HR)
        sam = SAM(SR, HR)
        
        PSNRs.append(psnr)
        SSIMs.append(ssim)
        SAMs.append(sam)
        
        SR = SR.transpose(1,2,0)   
        HR = HR.transpose(1,2,0)  
        
	                    
        scio.savemat(out_path + images_name[index], {'HR': HR, 'SR':SR})  
        print("===The {}-th picture=====PSNR:{:.3f}=====SSIM:{:.4f}=====SAM:{:.3f}====Name:{}".format(index+1,  psnr, ssim, sam, images_name[index]))                 
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(np.mean(PSNRs), np.mean(SSIMs), np.mean(SAMs))) 
    print(T/len(images_name))
if __name__ == "__main__":
    main()
