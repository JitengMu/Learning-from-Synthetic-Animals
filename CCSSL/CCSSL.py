from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import _init_paths
from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds,  calc_metrics, get_preds
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
import pose.losses as losses

import scripts.ssl_datasets as datasets
from options.train_options import TrainOptions
from CCSSL.scripts.timer import Timer
from CCSSL.scripts.consistency import prediction_check

opt = TrainOptions()
args = opt.initialize()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
idx = range(1, args.num_classes+1)
njoints = len(idx)
criterion = losses.JointsMSELoss().to(device)
global_animal = args.animal

def main():
    
    _t = {'iter time' : Timer()}

    # create directory
    model_name = args.source + '_to_' + args.target
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
        os.makedirs(os.path.join(args.checkpoint, 'logs'))
        os.makedirs(os.path.join(args.checkpoint, 'ssl_labels'))
    opt.print_options(args)

    # load datasets
    # load synthetic datasets
    source_train_dataset = datasets.__dict__[args.source](is_train=True, **vars(args))
    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    source_test_dataset = datasets.__dict__[args.source](is_train=False, **vars(args))
    source_test_loader = torch.utils.data.DataLoader(
        source_test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    # load real training set images with pseudo-labels
    target_train_dataset = datasets.__dict__[args.target_ssl](is_train=True, is_aug=False, **vars(args))
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    # load original real test set with ground truth labels
    target_test_dataset = datasets.__dict__[args.target](is_train=False, is_aug=False, **vars(args))
    target_test_loader = torch.utils.data.DataLoader(
        target_test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # create model and optimizer    
    model, optimizer = CreateModel(args, models, datasets)
 
    cudnn.enabled = True
    cudnn.benchmark = True
    model.train()
    model = torch.nn.DataParallel(model).to(device)
    if isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    loss = ['loss_kpt_src', 'loss_kpt_trg']
    _t['iter time'].tic()
    trg_val_acc_best = 0

    # CC-SSL training
    for epoch in range(args.num_epochs):
        if epoch==0:
            trg_val_loss, trg_val_acc = validate(target_test_loader, model, criterion, args.flip, args.batch_size, njoints)

        # generate ssl labels every 10 epoch
        if epoch%10==0:
            print("==> generating ssl labels")

            target_train_dataset = datasets.__dict__[args.target](is_train=True, is_aug=False, **vars(args))
            target_train_loader = torch.utils.data.DataLoader(
                target_train_dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True
            )

            model.eval()

            # generate labels on target training set
            ssl_kpts = {}
            acces1 = AverageMeter()
            previous_img = None
            previous_kpts = None
            for _, (trg_img, trg_lbl, trg_meta) in enumerate(target_train_loader):
                trg_img = trg_img.to(device)
                trg_lbl = trg_lbl.to(device, non_blocking=True)
                # generate labels for each image
                for i in range(trg_img.size(0)):
                    score_map, generated_kpts = prediction_check(previous_img, previous_kpts, trg_img[i], model, target_train_dataset)
                    ssl_kpts[int(trg_meta['index'][i].cpu().numpy().astype(np.int32))] = generated_kpts
                    if global_animal=='tiger':
                        trg_lbl[i] = trg_lbl[i,np.array([1,2,3,4,5,6,7,8,15,16,17,18,13,14,9,10,11,12])-1,:,:]

                    acc1, _ = accuracy(score_map, trg_lbl[i].cpu().unsqueeze(0), idx)
                    acces1.update(acc1[0], 1)
                    previous_img = trg_img[i]
                    previous_kpts = generated_kpts
            print('Acc on target training set (pseudo-labels):', acces1.avg)

            # modify confidence score based on ranking
            sorted_confidence = np.zeros(1)
            for k in ssl_kpts:
                sorted_confidence = np.concatenate( (sorted_confidence, ssl_kpts[k][:,2].reshape(-1)), axis=0)
            sorted_confidence = np.sort(sorted_confidence)
            np.save(args.checkpoint + '/ssl_labels/ssl_labels_train_confidence.npy', sorted_confidence)
            p = (1.0-0.02*(epoch+10))
            if p>0.2:
                ccl_thresh = sorted_confidence[int( p * sorted_confidence.shape[0])]
            else:
                p = 0.2
                ccl_thresh = sorted_confidence[int( p * sorted_confidence.shape[0])]
            print("=====> ccl_thresh: ", ccl_thresh)
            for k in ssl_kpts:
                ssl_kpts[k][:,2] = (ssl_kpts[k][:,2]>ccl_thresh).astype(np.float32)


            np.save(args.checkpoint + 'ssl_labels/ssl_labels_train.npy', ssl_kpts)

            # generate labels on target test set for diagnosis
            ssl_kpts = {}
            acces1 = AverageMeter()
            previous_img = None
            previous_kpts = None
            for jj, (trg_img, trg_lbl, trg_meta) in enumerate(target_test_loader):
                trg_img = trg_img.to(device)
                trg_lbl = trg_lbl.to(device, non_blocking=True)
                # generate labels for each image
                for i in range(trg_img.size(0)):
                    score_map, generated_kpts = prediction_check(previous_img, previous_kpts, trg_img[i], model, target_test_dataset)
                    ssl_kpts[int(trg_meta['index'][i].cpu().numpy().astype(np.int32))] = generated_kpts
                    if global_animal=='tiger':
                        trg_lbl[i] = trg_lbl[i,np.array([1,2,3,4,5,6,7,8,15,16,17,18,13,14,9,10,11,12])-1,:,:]

                    acc1, _ = accuracy(score_map, trg_lbl[i].cpu().unsqueeze(0), idx)
                    acces1.update(acc1[0], 1)
                    previous_img = trg_img[i]
                    previous_kpts = generated_kpts
            print('Acc on target testing set (pseudo-labels):', acces1.avg)
            np.save(args.checkpoint + 'ssl_labels/ssl_labels_valid'+str(epoch)+'.npy', ssl_kpts)

            # load real training set images with pseudo-labels
            target_train_dataset = datasets.__dict__[args.target_ssl](is_train=True, is_aug=True, **vars(args))
            target_train_loader = torch.utils.data.DataLoader(
                target_train_dataset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True
            )
            print("======> start training")
 
        loss_output_src = AverageMeter()
        loss_output_trg = AverageMeter()

        joint_loader = zip(source_train_loader, target_train_loader)
        model.train()

        # training with source images and target images jointly
        for i, ((src_img, src_lbl, src_meta), (trg_img, trg_lbl, trg_meta)) in enumerate(joint_loader):

            optimizer.zero_grad()

            # calculate loss on source dataset
            src_img, src_lbl, src_weight = src_img.to(device), src_lbl.to(device, non_blocking=True), src_meta['target_weight'].to(device, non_blocking=True)
            src_kpt_score = model(src_img)
            if type(src_kpt_score) == list:  # multiple output
                loss_kpt_src = 0
                for o in src_kpt_score:
                    loss_kpt_src += criterion(o, src_lbl, src_weight, len(idx))
                src_kpt_score = src_kpt_score[-1]
            else:  # single output
                loss_kpt_src = criterion(src_kpt_score, src_lbl, src_weight, len(idx))
            loss_output_src.update(loss_kpt_src.data.item(), src_img.size(0))
            loss_kpt_src.backward()
 
            # calculate loss on target dataset
            trg_img, trg_lbl, trg_weight = trg_img.to(device), trg_lbl.to(device, non_blocking=True), trg_meta['target_weight'].to(device, non_blocking=True)
            trg_kpt_score = model(trg_img)
            if type(trg_kpt_score) == list:  # multiple output
                loss_kpt_trg = 0
                for o in trg_kpt_score:
                    loss_kpt_trg += criterion(o, trg_lbl, trg_weight, len(idx))
                trg_kpt_score = trg_kpt_score[-1]
            else:  # single output
                loss_kpt_trg = criterion(trg_kpt_score, trg_lbl, trg_weight, len(idx))
            loss_kpt_trg *= args.gamma_
            loss_output_trg.update(loss_kpt_trg.data.item(), src_img.size(0))
            loss_kpt_trg.backward()

            # update
            optimizer.step()

            # print logs
            if (i+1) % args.print_freq == 0:
                _t['iter time'].toc(average=False)
                print('[epoch %d][it %d][src kpt loss %.6f][trg kpt loss %.6f][lr %.6f][%.2fs]' % \
                    (epoch+1, i + 1, loss_output_src.avg, loss_output_trg.avg, optimizer.param_groups[0]['lr'], _t['iter time'].diff))
                _t['iter time'].tic()

        print('\nEvaluation')
        src_val_loss, src_val_acc = validate(source_test_loader, model, criterion, args.flip, args.batch_size, njoints)
        trg_val_loss, trg_val_acc = validate(target_test_loader, model, criterion, args.flip, args.batch_size, njoints)

        # save best model
        if trg_val_acc>trg_val_acc_best:
            trg_val_acc_best = trg_val_acc
            print('\ntaking snapshot ...')
            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, os.path.join(args.checkpoint, '%s' %(args.source) +'.pth.tar' ))

def validate(val_loader, model, criterion, flip=True, test_batch=6, njoints=18):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), njoints, 2)

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))

    with torch.no_grad():
        for i, (input, target, meta) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(device, non_blocking=True)
            if global_animal=='horse':
                target = target.to(device, non_blocking=True)
                target_weight = meta['target_weight'].to(device, non_blocking=True)
            elif global_animal=='tiger':
                target = target.to(device, non_blocking=True)
                target_weight = meta['target_weight'].to(device, non_blocking=True)
                target = target[:,np.array([1,2,3,4,5,6,7,8,15,16,17,18,13,14,9,10,11,12])-1,:,:]
                target_weight = target_weight[:,np.array([1,2,3,4,5,6,7,8,15,16,17,18,13,14,9,10,11,12])-1,:]
            else:
                raise Exception('please add new animal category')

            # compute output
            output = model(input)
            score_map = output[-1].cpu() if type(output) == list else output.cpu()
            if flip:
                flip_input = torch.from_numpy(fliplr(input.clone().numpy())).float().to(device)
                flip_output = model(flip_input)
                flip_output = flip_output[-1].cpu() if type(flip_output) == list else flip_output.cpu()
                flip_output = flip_back(flip_output)
                score_map += flip_output

            if type(output) == list:  # multiple output
                loss = 0
                for o in output:
                    loss += criterion(o, target, target_weight, len(idx))
                output = output[-1]
            else:  # single output
                loss = criterion(output, target, target_weight, len(idx))

            acc, _ = accuracy(score_map, target.cpu(), idx)

            # generate predictions
            preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
            #for n in range(score_map.size(0)):
            #    predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            acces.update(acc[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.8f} | Acc: {acc: .8f}'.format(
                        batch=i + 1,
                        size=len(val_loader),
                        data=data_time.val,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        acc=acces.avg
                        )
            bar.next()

        bar.finish()
    return losses.avg, acces.avg

def CreateModel(args, models, datasets):
    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    njoints = datasets.__dict__[args.source].njoints
    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__['hg'](num_stacks=args.stacks,
                                   num_blocks=args.blocks,
                                   num_classes=njoints,
                                   resnet_layers=args.resnet_layers)

    #model = torch.nn.DataParallel(model).to(device)
    optimizer = optim.RMSprop(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    return model, optimizer


 
if __name__ == '__main__':
    torch.set_printoptions(precision=16)
    main()
    
    
        
