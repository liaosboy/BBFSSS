from util.dataset import BBPFEDataset
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU
from util import transform
from models.fewshot import FewShotSeg

import torch.nn as nn
import torch.nn.functional as F
import util.config as config
import torch.utils.data as data
import numpy as np
import torch
import os
import random
import time
import cv2


class Train:
    def __init__(self) -> None:
        pass

    def main(self):
        value_scale = 255
        mean = config.mean
        mean = [item * value_scale for item in mean]
        std = config.std
        std = [item * value_scale for item in std]

        train_sup_transform = [
            transform.RandomGaussianBlur(),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)]
        train_sup_transform = transform.Compose(train_sup_transform)

        train_que_transform = [
            # transform.RandScale([config.scale_min, config.scale_max]),
            transform.RandRotate([config.rotate_min, config.rotate_max], padding=mean, ignore_label=config.padding_label),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)]
        train_que_transform = transform.Compose(train_que_transform)

        train_dataset = BBPFEDataset(config.data_root, config.classes,
                split= config.split,shot= config.shot, mode = "train", fold_num=config.fold, img_size=config.img_size, que_transform=train_que_transform, sup_transform=train_sup_transform)
        
        train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )

        val_que_transform = [
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)]
        val_que_transform = transform.Compose(val_que_transform)

        val_dataset = BBPFEDataset(config.data_root, config.classes,
                split= config.split,shot= config.shot, mode = "val", fold_num=config.fold, img_size=config.img_size, que_transform=val_que_transform, sup_transform=val_que_transform)
        
        val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False
        )



        model = FewShotSeg(pretrained_path= config.init_path, cfg=config.model)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.base_lr, momentum=config.momentum, weight_decay=config.weight_decay)


        print("Start training")
        max_iou = 0.
        filename = 'BBFSSS.pth'
        for epoch in range(config.epochs):
            print(f"Train Epoch : {epoch+1} / {config.epochs}")
            if config.fix_random_seed_val:
                torch.cuda.manual_seed(config.manual_seed + epoch)
                np.random.seed(config.manual_seed + epoch)
                torch.manual_seed(config.manual_seed + epoch)
                torch.cuda.manual_seed_all(config.manual_seed + epoch)
                random.seed(config.manual_seed + epoch)

            self.train(loss_fn, train_loader, model, optimizer, epoch)
            
            if epoch % 2 == 0 :
                loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = self.val(val_loader, model)
                if class_miou > max_iou:
                    max_iou = class_miou
                    filename = config.save_path + '/train_epoch_' + str(epoch) + '_'+str(max_iou)+'.pth'
                    if os.path.exists(filename):
                        os.remove(filename)            
                    print('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

    def train(self, loss_fn, dataloader:data.DataLoader, model:nn.Module, optimizer, epoch):
        loss_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        label_meter = AverageMeter()

        model.train()
        max_iter = config.epochs * len(dataloader)
        for i, (image, label, sup_img, sup_bblabel, sup_semlabel, _) in enumerate(dataloader):
            if config.base_lr > 1e-6:
                current_iter = epoch * len(dataloader) + i + 1
                poly_learning_rate(optimizer, config.base_lr, current_iter, max_iter, power=config.power)
            temp = label[0]

            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            sup_img = sup_img.cuda(non_blocking=True)
            sup_bblabel = sup_bblabel.cuda(non_blocking=True)
            # sup_semlabel = sup_semlabel.cuda(non_blocking=True)

            output, new_label = model(s_x=sup_img, s_y=sup_bblabel, q_x=image, q_y=label)
            _, pred = torch.max(output, 1)

            loss = loss_fn(output, new_label)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            n = image.size(0)

            intersection, union, label = intersectionAndUnionGPU(pred, label, 2, 255)
            intersection, union, label = intersection.cpu().numpy(), union.cpu().numpy(), label.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), label_meter.update(label)

            accuracy = sum(intersection_meter.val) / (sum(label_meter.val) + 1e-10)
            loss_meter.update(loss.item(), n)

            if (i+1) % 20 == 0:
                print(f"Train : Epoch {epoch + 1}, Batch [{i+1} / {len(dataloader)}], Loss : {'%.4f'%loss.data}, Acc : {accuracy}")

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (label_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(label_meter.sum) + 1e-10)
        print('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, config.epochs, mIoU, mAcc, allAcc))

    def val(self, dataloader:data.DataLoader, model:nn.Module):
        print('Start Evaluation')
        batch_time = AverageMeter()
        model_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        criterion = nn.CrossEntropyLoss()
        split_gap = int(len(config.classes) / config.fold)
        class_intersection_meter = [0]*split_gap
        class_union_meter = [0]*split_gap
        
        if config.manual_seed is not None and config.fix_random_seed_val:
            torch.cuda.manual_seed(config.manual_seed)
            np.random.seed(config.manual_seed)
            torch.manual_seed(config.manual_seed)
            torch.cuda.manual_seed_all(config.manual_seed)
            random.seed(config.manual_seed)

        model.eval()
        end = time.time()
        test_num = len(dataloader)
        
        iter_num = 0
        total_time = 0


        for i, (image, label, sup_img, sup_bblabel, sup_semlabel, cls) in enumerate(dataloader):
            if (iter_num-1) >= test_num:
                break
            iter_num += 1

            data_time.update(time.time() - end)
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            sup_img = sup_img.cuda(non_blocking=True)
            sup_bblabel = sup_bblabel.cuda(non_blocking=True)
            # sup_semlabel = sup_semlabel.cuda(non_blocking=True)
            start_time = time.time()

            output = model(s_x=sup_img, s_y=sup_bblabel, x=image, y=label)

            total_time = total_time + 1
            model_time.update(time.time() - start_time) 

            output = F.interpolate(output, size=label.size()[1:], mode='bilinear', align_corners=True)         
            loss = criterion(output, label)              

            # n = image.size(0)
            loss = torch.mean(loss)

            output = output.max(1)[1]

            intersection, union, new_target = intersectionAndUnionGPU(output, label, 2, 255)
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), label.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
                
            subcls = int(cls[0].cpu().numpy()[0])
            class_intersection_meter[(subcls)%split_gap] += intersection[1]
            class_union_meter[(subcls)%split_gap] += union[1] 

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), image.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % 20 == 0):
                print('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num, test_num,
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            loss_meter=loss_meter,
                                                            accuracy=accuracy))

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        
        class_iou_class = []
        class_miou = 0
        for i in range(len(class_intersection_meter)):
            class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
            class_iou_class.append(class_iou)
            class_miou += class_iou
        class_miou = class_miou*1.0 / len(class_intersection_meter)
        print('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
        for i in range(split_gap):
            print('Class_{} Result: iou {:.4f}.'.format(i+1, class_iou_class[i]))            
        

        
        print('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(2):
            print('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

        print('avg inference time: {:.4f}, count: {}'.format(model_time.avg, test_num))
        return loss_meter.avg, mIoU, mAcc, allAcc, class_miou


if __name__ == "__main__":
    trainer = Train()
    trainer.main()