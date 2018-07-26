import time
import torch
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy


def validate(dataset, model, criterion, epoch, device, args):
    """
    Evaluates/validates the model

    Parameters:
        dataset (torch.utils.data.TensorDataset): The dataset
        model (torch.nn.module): Model to be evaluated/validated
        criterion (torch.nn.criterion): Loss function
        epoch (int): Epoch counter
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int).

    Returns:
        float: top1 accuracy
    """
    cl_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(dataset.val_loader):
            input, target = input.to(device), target.to(device)

            # compute output
            output = model(input)

            # make targets one-hot for using BCEloss
            target_temp = target
            one_hot = torch.zeros(target.size(0), output.size(1)).to(device)
            one_hot.scatter_(1, target.long().view(target.size(0), -1), 1)
            target = one_hot

            # compute loss and accuracy
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output, target_temp, (1,5))

            # measure accuracy and record loss
            cl_losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

    #
    print(' * Validation Task: \nLoss {loss.avg:.5f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(loss=cl_losses, top1 = top1, top5 = top5))
    print('=' * 80)

    return top1.avg

