from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

import _init_paths
from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds,  calc_metrics, get_preds
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap, draw_labelmap
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
import pose.datasets as datasets
import pose.losses as losses

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def inter_and_union(pred, mask, num_class=10):
  pred = np.asarray(pred, dtype=np.uint8).copy()
  mask = np.asarray(mask, dtype=np.uint8).copy()

  inter = pred * (pred == mask)
  (area_inter, _) = np.histogram(inter, bins=num_class-1, range=(1, num_class))
  (area_pred, _) = np.histogram(pred, bins=num_class-1, range=(1, num_class))
  (area_mask, _) = np.histogram(mask, bins=num_class-1, range=(1, num_class))
  area_union = area_pred + area_mask - area_inter

  return (area_inter, area_union)

# get model names and dataset names
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


dataset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))


# init global variables
best_acc = 0
best_iou = 0
best_epoch_acc = 0
best_epoch_iou = 0
idx = []

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33

def main(args):
    global best_acc
    global best_iou
    global best_epoch_acc
    global best_epoch_iou
    global idx

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    njoints = datasets.__dict__[args.dataset].njoints

    # idx is the index of joints used to compute accuracy
    idx = range(1,njoints+1)

    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks,
                                       num_blocks=args.blocks,
                                       num_classes=njoints,
                                       resnet_layers=args.resnet_layers)

    model = torch.nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = losses.JointsMSELoss().to(device)
    criterion_seg = nn.CrossEntropyLoss(ignore_index=255).to(device)

    if args.solver == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    elif args.solver == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )
    else:
        print('Unknown solver: {}'.format(args.solver))
        assert False

    # optionally resume from a checkpoint
    title = args.dataset + ' ' + args.arch
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss',
                          'Train Acc', 'Val Acc', 'Val IOU'])

    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters())/1000000.0))

    # create data loader
    train_dataset = datasets.__dict__[args.dataset](is_train=True, **vars(args))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    val_dataset = datasets.__dict__[args.dataset](is_train=False, **vars(args))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # evaluation only
    if args.evaluate:
        print('\nEvaluation only')
        loss, acc, predictions, auc, mean_error = validate(val_loader, model, criterion,
                                          args.debug, args.flip, args.test_batch, njoints)
        save_pred(predictions, checkpoint=args.checkpoint)
        return

    # train and eval
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *=  args.sigma_decay
            val_loader.dataset.sigma *=  args.sigma_decay

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, criterion_seg, optimizer,
                                      args.debug, args.flip, args.train_batch, epoch, njoints)

        # evaluate on validation set
        valid_loss, valid_acc, predictions, valid_iou = validate(val_loader, model, criterion, criterion_seg,
                                                  args.debug, args.flip, args.test_batch, njoints)

        # append logger file
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc, valid_iou])

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        #is_best = valid_auc>best_auc
        if valid_acc>best_acc:
            best_epoch_acc = epoch
        if valid_iou>best_iou:
            best_epoch_iou = epoch

        best_acc = max(valid_acc, best_acc)
        best_iou = max(valid_iou, best_iou)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, predictions, is_best, checkpoint=args.checkpoint, snapshot=args.snapshot)

    print(best_epoch_acc, best_acc, best_epoch_iou, best_iou)
    logger.close()
    #logger.plot(['Train Acc', 'Val Acc'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(train_loader, model, criterion, criterion_seg, optimizer, debug=False, flip=True, train_batch=6, epoch=0, njoints=68):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_kpt = AverageMeter()
    losses_seg = AverageMeter()
    acces = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    gt_win, pred_win = None, None
    bar = Bar('Train', max=len(train_loader))

    for i, (input, target, target_seg, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target, target_seg = input.to(device), target.to(device, non_blocking=True), target_seg.to(device)
        target_weight = meta['target_weight'].to(device, non_blocking=True)

        # compute output
        output_kpt, output_seg = model(input)
        if type(output_kpt) == list:  # multiple output
            loss_kpt = 0
            loss_seg = 0
            for (o, o_seg) in zip(output_kpt, output_seg):
                loss_kpt += criterion(o, target, target_weight, len(idx))
                loss_seg += criterion_seg(o_seg, target_seg)
            output_kpt = output_kpt[-1]
            output_seg = output_seg[-1]
        else:  # single output
            loss_kpt = criterion(output_kpt, target, target_weight, len(idx))
            loss_seg = criterion_seg(output_seg, target_seg)
        acc, batch_interocular_dists = accuracy(output_kpt, target, idx)
        _, pred_seg = torch.max(output_seg, 1)

        if debug: # visualize groundtruth and predictions
            gt_batch_img = batch_with_heatmap(input, target)
            pred_batch_img = batch_with_heatmap(input, output)
            if not gt_win or not pred_win:
                ax1 = plt.subplot(121)
                ax1.title.set_text('Groundtruth')
                gt_win = plt.imshow(gt_batch_img)
                ax2 = plt.subplot(122)
                ax2.title.set_text('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        loss = loss_kpt + (0.01/(epoch+1))*loss_seg

        # measure accuracy and record loss
        losses_kpt.update(loss_kpt.item(), input.size(0))
        losses_seg.update(loss_seg.item(), input.size(0))
        acces.update(acc[0], input.size(0))
        
        inter, union = inter_and_union(pred_seg.data.cpu().numpy().astype(np.uint8), target_seg.data.cpu().numpy().astype(np.uint8))
        inter_meter.update(inter)
        union_meter.update(union)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        iou = inter_meter.sum / (union_meter.sum + 1e-10)

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_kpt: {loss_kpt:.8f} | Loss_seg: {loss_seg:.8f} | Acc: {acc: .4f} | IOU: {iou: .2f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss_kpt=losses_kpt.avg,
                    loss_seg=losses_seg.avg,
                    acc=acces.avg,
                    iou=iou.mean()*100
                    )
        bar.next()

    bar.finish()
    print(iou)
    return losses_kpt.avg, acces.avg


def validate(val_loader, model, criterion, criterion_seg, debug=False, flip=True, test_batch=6, njoints=68):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_kpt = AverageMeter()
    losses_seg = AverageMeter()
    acces = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), njoints, 2)

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))

    interocular_dists = torch.zeros((njoints, val_loader.dataset.__len__()))

    with torch.no_grad():
        for i, (input, target, target_seg, meta) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input, target, target_seg = input.to(device), target.to(device, non_blocking=True), target_seg.to(device)
            target_weight = meta['target_weight'].to(device, non_blocking=True)

            # compute output
            output_kpt, output_seg = model(input)
            score_map = output_kpt[-1].cpu() if type(output_kpt) == list else output_kpt.cpu()
            
            if flip:
                flip_input = torch.from_numpy(fliplr(input.clone().numpy())).float().to(device)
                flip_output = model(flip_input)
                flip_output = flip_output[-1].cpu() if type(flip_output) == list else flip_output.cpu()
                flip_output = flip_back(flip_output)
                score_map += flip_output

            if type(output_kpt) == list:  # multiple output
                loss_kpt = 0
                loss_seg = 0
                for (o, o_seg) in zip(output_kpt, output_seg):
                    loss_kpt += criterion(o, target, target_weight, len(idx))
                    loss_seg += criterion_seg(o_seg, target_seg)
                output = output_kpt[-1]
                output_seg = output_seg[-1]
            else:  # single output
                loss_kpt = criterion(output_kpt, target, target_weight, len(idx))
                loss_seg = criterion(output_seg, target_seg)

            acc, batch_interocular_dists = accuracy(score_map, target.cpu(), idx)
            _, pred_seg = torch.max(output_seg, 1) 

            # generate predictions
            preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            if debug:
                gt_batch_img = batch_with_heatmap(input, target)
                pred_batch_img = batch_with_heatmap(input, score_map)
                if not gt_win or not pred_win:
                    plt.subplot(121)
                    gt_win = plt.imshow(gt_batch_img)
                    plt.subplot(122)
                    pred_win = plt.imshow(pred_batch_img)
                else:
                    gt_win.set_data(gt_batch_img)
                    pred_win.set_data(pred_batch_img)
                plt.pause(.05)
                plt.draw()

            # measure accuracy and record loss
            losses_kpt.update(loss_kpt.item(), input.size(0))
            losses_seg.update(loss_seg.item(), input.size(0))
            acces.update(acc[0], input.size(0))

            inter, union = inter_and_union(pred_seg.data.cpu().numpy().astype(np.uint8), target_seg.data.cpu().numpy().astype(np.uint8))
            inter_meter.update(inter)
            union_meter.update(union)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            iou = inter_meter.sum / (union_meter.sum + 1e-10)

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_kpt: {loss_kpt:.8f} | Loss_seg: {loss_seg:.8f} | Acc: {acc: .8f} | IOU: {iou:.2f}'.format(
                        batch=i + 1,
                        size=len(val_loader),
                        data=data_time.val,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss_kpt=losses_kpt.avg,
                        loss_seg=losses_seg.avg,
                        acc=acces.avg,
                        iou=iou.mean()*100
                        )
            bar.next()

        bar.finish()
        print(iou)
    return losses_kpt.avg, acces.avg, predictions, iou.mean()*100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset setting
    parser.add_argument('--dataset', metavar='DATASET', default='synthetic_animal_sp_multitask',
                        choices=dataset_names,
                        help='Datasets: ' +
                            ' | '.join(dataset_names) +
                            ' (default: mpii)')
    parser.add_argument('--image-path', default='./animal_data/', type=str,
                       help='path to images')
    parser.add_argument('--animal', default='horse', type=str,
                       help='horse | tiger')
    #parser.add_argument('--anno-path', default='./data/mpii/mpii_annotations.json', type=str,
    #                    help='path to annotation (json)')
    parser.add_argument('--year', default=2014, type=int, metavar='N',
                        help='year of coco dataset: 2014 (default) | 2017)')
    parser.add_argument('--inp-res', default=256, type=int,
                        help='input resolution (default: 256)')
    parser.add_argument('--out-res', default=64, type=int,
                    help='output resolution (default: 64, to gen GT)')

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: hg)')
    parser.add_argument('-s', '--stacks', default=8, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('--resnet-layers', default=50, type=int, metavar='N',
                        help='Number of resnet layers',
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    # Training strategy
    parser.add_argument('--solver', metavar='SOLVER', default='rms',
                        choices=['rms', 'adam'],
                        help='optimizers')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--target-weight', dest='target_weight',
                        action='store_true',
                        help='Loss with target_weight')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--scale-factor', type=float, default=0.25,
                        help='Scale factor (data aug).')
    parser.add_argument('--rot-factor', type=float, default=30,
                        help='Rotation factor (data aug).')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--snapshot', default=0, type=int,
                        help='save models for every #snapshot epochs (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')


    main(parser.parse_args())
