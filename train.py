import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
from option import opt
from crossformer11 import CrossFormer
from data_utils import TrainsetFromFolder, ValsetFromFolder
from eval import PSNR, SSIM, SAM
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

import scipy.io as scio

psnr = []
out_path = './result/'


def main():
    if opt.show:
        global writer
        writer = SummaryWriter(log_dir='logs')

    if opt.cuda:
        print("=> Use GPU ID: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    # Loading datasets
    # train_set = TrainsetFromFolder('/media/hdisk/liqiang/hyperSR/train/'+ opt.datasetName + '/' +  str(opt.upscale_factor) + '/')
    train_set = TrainsetFromFolder('D:/study/数据集/CAVE处理后/×2/train')
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    # val_set = ValsetFromFolder('/media/hdisk/liqiang/hyperSR/test/' + opt.datasetName + '/' + str(opt.upscale_factor))
    val_set = ValsetFromFolder('D:/study/数据集/CAVE处理后/×2/test/')
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    # Buliding model
    model = CrossFormer(opt)
    criterion = nn.L1Loss()

    if opt.cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()
    print('# parameters:', sum(param.numel() for param in model.parameters()))

    # Setting Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)

    # optionally resuming from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)  
            check_epoch = checkpoint['epoch']
            opt.start_epoch = check_epoch + 1 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler = MultiStepLR(optimizer, milestones=[35, 70, 105, 140, 175], gamma=0.5, last_epoch = -1)
            # print(check_epoch)
            # for i in range(check_epoch):
             #     scheduler.step()
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

            # Setting learning rate
    scheduler = MultiStepLR(optimizer, milestones=[35, 70, 105, 140, 175], gamma=0.5, last_epoch=-1)

    # Training
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        scheduler.step()
        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
        train(train_loader, optimizer, model, criterion, epoch)
        val(val_loader, model.module, epoch)
        save_checkpoint(model, epoch, optimizer)


def train(train_loader, optimizer, model, criterion, epoch):
    model.train()

    for iteration, batch in enumerate(train_loader, 1):
        input, label = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        if opt.cuda:
            input = input.cuda()
            label = label.cuda()

        localFeats = []
        for i in range(11):
            
            if i == 0:
                x = input[:, 0:5, :, :]
                # y = input[:,0,:,:]
                new_label = label[:, 0:2, :, :]
            elif i == 1:
                x = input[:, 1:6, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 2:5, :, :]
            elif i == 2:
                x = input[:, 4:9, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 5:8, :, :]
            elif i == 3:
                x = input[:, 7:12, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 8:11, :, :]
            elif i == 4:
                x = input[:, 10:15, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 11:14, :, :]
            elif i == 5:
                x = input[:, 13:18, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 14:17, :, :]
            elif i == 6:
                x = input[:, 16:21, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 17:20, :, :]
            elif i == 7:
                x = input[:, 19:24, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 20:23, :, :]
            elif i == 8:
                x = input[:, 22:27, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 23:26, :, :]

            elif i == 9:
                x = input[:, 25:30, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 26:29, :, :]
            elif i == 10:
                x = input[:, 26:31, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 29:31, :, :]

            SR, localFeats = model(x, localFeats, i)
            localFeats.detach_()
            localFeats = localFeats.detach()
            localFeats = Variable(localFeats.data, requires_grad=False)
            loss = criterion(SR, new_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if iteration % 100 == 0:
            print("{}===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
                time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())), epoch, iteration, len(train_loader),
                loss.item()))

        if opt.show:
            writer.add_scalar('Train/Loss', loss.data[0])


def val(val_loader, model, epoch):
    model.eval()
    val_psnr = 0
    val_ssim = 0
    val_sam = 0

    for iteration, batch in enumerate(val_loader, 1):
        with torch.no_grad():
          input, label = Variable(batch[0]), Variable(batch[1])
          SR = np.ones((label.shape[1], label.shape[2], label.shape[3])).astype(np.float32)
  
          if opt.cuda:
              input = input.cuda()
          localFeats = []
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
                output,localFeats = model(x, localFeats, i)
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
          val_psnr += PSNR(SR, label.data[0].numpy())
          val_ssim += SSIM(SR, label.data[0].numpy())
          val_sam += SAM(SR, label.data[0].numpy())

    val_psnr = val_psnr / len(val_loader)
    val_ssim = val_ssim / len(val_loader)
    val_sam = val_sam / len(val_loader)

    print("PSNR = {:.3f}, SSIM = {:.4f}, SAM = {:.3f}".format(val_psnr, val_ssim, val_sam))
    if opt.show:
        writer.add_scalar('Val/PSNR', val_psnr, epoch)


def save_checkpoint(model, epoch, optimizer):
    model_out_path = "checkpoint/" + "model_{}_epoch_{}.pth".format(opt.upscale_factor, epoch)
    state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    if not os.path.exists("checkpointgo/"):
        os.makedirs("checkpointgo/")
    torch.save(state, model_out_path)


if __name__ == "__main__":
    main()