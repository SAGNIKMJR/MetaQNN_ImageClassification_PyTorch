import time
import torch
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy

def train(dataset, model, criterion, epoch, optimizer, lr_scheduler, device, args):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        train_loader (torch.utils.data.DataLoader): The trainset dataloader
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        lr_scheduler (Training.LearningRateScheduler): class implementing learning rate schedules
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain learning_rate (float), momentum (float),
            weight_decay (float), nesterov momentum (bool), lr_dropstep (int),
            lr_dropfactor (float), print_freq (int) and expand (bool).
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    cl_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(dataset.train_loader):
        input, target = input.to(device), target.to(device)
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust the learning rate if applicable
        lr_scheduler.adjust_learning_rate(optimizer, i + 1)

        # compute output
        output = model(input)

        # making targets one-hot for using BCEloss
        target_temp = target
        one_hot = torch.zeros(target.size(0), output.size(1)).to(device)
        one_hot.scatter_(1, target.long().view(target.size(0), -1), 1)
        target = one_hot

        # compute loss and accuracy
        loss = criterion(output, target)
        prec1, prec5 = accuracy(output, target_temp, (1,5))

        # measure accuracy and record loss
        cl_losses.update(loss.item(), input.size(0))
        prec1, prec5 = accuracy(output, target_temp, (1,5))

        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del output, input, target
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                   epoch, i, len(dataset.train_loader), batch_time=batch_time,
                   data_time=data_time, loss=cl_losses, top1=top1, top5 = top5))


    lr_scheduler.scheduler_epoch += 1

    print(' * Train: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    print('=' * 80)
