import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from conf import get_config, set_env, set_logger, set_outdir
from dataset import *
from model.MEFL import MEFARG
from utils import *


def get_dataloader(conf):
    print("==> Preparing data...")
    if conf.dataset == "BP4D":
        trainset = BP4D(
            conf.dataset_path,
            train=True,
            fold=conf.fold,
            transform=image_train(crop_size=conf.crop_size),
            crop_size=conf.crop_size,
            stage=2,
        )
        train_loader = DataLoader(
            trainset,
            batch_size=conf.batch_size,
            shuffle=True,
            num_workers=conf.num_workers,
        )
        valset = BP4D(
            conf.dataset_path,
            train=False,
            fold=conf.fold,
            transform=image_test(crop_size=conf.crop_size),
            stage=2,
        )
        val_loader = DataLoader(
            valset,
            batch_size=conf.batch_size,
            shuffle=False,
            num_workers=conf.num_workers,
        )

    elif conf.dataset == "DISFA":
        trainset = DISFA(
            conf.dataset_path,
            train=True,
            fold=conf.fold,
            transform=image_train(crop_size=conf.crop_size),
            crop_size=conf.crop_size,
            stage=2,
        )
        train_loader = DataLoader(
            trainset,
            batch_size=conf.batch_size,
            shuffle=True,
            num_workers=conf.num_workers,
        )
        valset = DISFA(
            conf.dataset_path,
            train=False,
            fold=conf.fold,
            transform=image_test(crop_size=conf.crop_size),
            stage=2,
        )
        val_loader = DataLoader(
            valset,
            batch_size=conf.batch_size,
            shuffle=False,
            num_workers=conf.num_workers,
        )

    return train_loader, val_loader, len(trainset), len(valset)


# Train
def train(conf, net, train_loader, optimizer, epoch, criterion):
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs, targets, relations) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(
            optimizer,
            epoch,
            conf.epochs,
            conf.learning_rate,
            batch_idx,
            train_loader_len,
        )
        targets, relations = targets.float(), relations.long()
        if torch.cuda.is_available():
            inputs, targets, relations = inputs.cuda(), targets.cuda(), relations.cuda()
        optimizer.zero_grad()
        outputs, outputs_relation = net(inputs)
        wa_loss = criterion[0](outputs, targets)
        edge_loss = criterion[1](outputs_relation.view(-1, 4), relations.view(-1))
        loss = wa_loss + conf.lam * edge_loss
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
        losses1.update(wa_loss.data.item(), inputs.size(0))
        losses2.update(edge_loss.data.item(), inputs.size(0))

    return losses.avg, losses1.avg, losses2.avg


# Val
def val(net, val_loader, criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        targets = targets.float()
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = net(inputs)
            loss = criterion[0](outputs, targets)
            losses.update(loss.data.item(), inputs.size(0))
            update_list = statistics(outputs, targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list


def main(conf):
    if conf.dataset == "BP4D":
        dataset_info = BP4D_infolist
    elif conf.dataset == "DISFA":
        dataset_info = DISFA_infolist

    start_epoch = 0
    # data
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)
    train_weight = torch.from_numpy(
        np.loadtxt(
            os.path.join(
                conf.dataset_path,
                "list",
                conf.dataset + "_weight_fold" + str(conf.fold) + ".txt",
            )
        )
    )
    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold + 1, conf.N_fold, val_data_num))
    net = MEFARG(num_classes=conf.num_classes, backbone=conf.arc)

    # resume
    if conf.resume != "":
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()

    criterion = [WeightedAsymmetricLoss(weight=train_weight), nn.CrossEntropyLoss()]
    optimizer = optim.AdamW(
        net.parameters(),
        betas=(0.9, 0.999),
        lr=conf.learning_rate,
        weight_decay=conf.weight_decay,
    )
    print("the init learning rate is ", conf.learning_rate)

    # train and val
    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]["lr"]
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        train_loss, wa_loss, edge_loss = train(conf, net, train_loader, optimizer, epoch, criterion)
        val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader, criterion)

        # log
        infostr = {
            "Epoch:  {}   train_loss: {:.5f} wa_loss: {:.5f} edge_loss: {:.5f} val_loss: {:.5f}  val_mean_f1_score {:.2f},val_mean_acc {:.2f}".format(
                epoch + 1,
                train_loss,
                wa_loss,
                edge_loss,
                val_loss,
                100.0 * val_mean_f1_score,
                100.0 * val_mean_acc,
            )
        }
        logging.info(infostr)
        infostr = {"F1-score-list:"}
        logging.info(infostr)
        infostr = dataset_info(val_f1_score)
        logging.info(infostr)
        infostr = {"Acc-list:"}
        logging.info(infostr)
        infostr = dataset_info(val_acc)
        logging.info(infostr)

        # save checkpoints
        if (epoch + 1) % 4 == 0:
            checkpoint = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                checkpoint,
                os.path.join(
                    conf["outdir"],
                    "epoch" + str(epoch + 1) + "_model_fold" + str(conf.fold + 1) + ".pth",
                ),
            )

        checkpoint = {
            "epoch": epoch,
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(
            checkpoint,
            os.path.join(conf["outdir"], "cur_model_fold" + str(conf.fold + 1) + ".pth"),
        )


# ---------------------------------------------------------------------------------


if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)
