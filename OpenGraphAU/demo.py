import logging
import os

import numpy as np
import torch
import torch.nn as nn

from conf import get_config, set_env, set_logger, set_outdir
from dataset import pil_loader
from model.ANFL import MEFARG
from OpenGraphAU.utils import *


def main(conf):
    dataset_info = hybrid_prediction_infolist

    # data
    img_path = conf.input

    net = MEFARG(
        num_main_classes=conf.num_main_classes,
        num_sub_classes=conf.num_sub_classes,
        backbone=conf.arc,
        neighbor_num=conf.neighbor_num,
        metric=conf.metric,
    )

    # resume
    if conf.resume != "":
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    net.eval()
    img_transform = image_eval()
    img = pil_loader(img_path)
    img_ = img_transform(img).unsqueeze(0)

    if torch.cuda.is_available():
        net = net.cuda()
        img_ = img_.cuda()

    with torch.no_grad():
        pred = net(img_)
        pred = pred.squeeze().cpu().numpy()

    # log
    infostr = {"AU prediction:"}
    logging.info(infostr)
    infostr_probs, infostr_aus = dataset_info(pred, 0.2)
    logging.info(infostr_aus)
    logging.info(infostr_probs)

    if conf.draw_text:
        img = draw_text(conf.input, list(infostr_aus), pred)
        import cv2

        path = conf.input.split(".")[0] + "_pred.jpg"
        cv2.imwrite(path, img)


# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)
