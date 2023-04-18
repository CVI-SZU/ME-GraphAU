import logging
import os

import cv2
import gdown
import pandas as pd
import seaborn as sns
import torch

from model.ANFL import MEFARG
from OpenGraphAU.model.face_detection import SCRFD_ONNX, ExpandBbox
from OpenGraphAU.utils import *

# Set the style and context of the plot
sns.set(rc={"figure.figsize": (25, 5)})
sns.set_style("whitegrid")
sns.set_context("notebook", rc={"lines.linewidth": 3})


class OpenGraphAU:
    def __init__(
        self,
        resume="checkpoints/OpenGprahAU-SwinB_first_stage.pth",
        num_main_classes=27,
        num_sub_classes=14,
        arc="swin_transformer_base",
        neighbor_num=4,
        metric="dots",
        device="cpu",
    ):
        self.device = torch.device(device)
        self.expand = ExpandBbox()
        self.transform = image_eval()
        self.face_detector = SCRFD_ONNX("checkpoints/face_detection/scrfd_500.onnx")
        self.net = MEFARG(
            num_main_classes=num_main_classes,
            num_sub_classes=num_sub_classes,
            backbone=arc,
            neighbor_num=neighbor_num,
            metric=metric,
        )

        # resume
        if resume != "":
            if not os.path.exists(resume):
                print("Downloading checkpoint: ", resume)
                if "SwinT" in resume:
                    gdown.download(
                        id="1JSa-ft965qXJlVGvnoMepbkRkSm78_to", output="checkpoints/OpenGprahAU-SwinT_first_stage.pth"
                    )
                if "SwinS" in resume:
                    gdown.download(
                        id="1GNjFKpd00nvgYIP2q7AzRSzfzEUfAqfT", output="checkpoints/OpenGprahAU-SwinS_first_stage.pth"
                    )
                if "SwinB" in resume:
                    gdown.download(
                        id="1nWwowmq4pQn1ACnSOOeyBy6-n0rmqTQ9", output="checkpoints/OpenGprahAU-SwinB_first_stage.pth"
                    )
                if "ResNet50" in resume:
                    gdown.download(
                        id="11xh9r2e4qCpWEtQ-ptJGWut_TQ0_AmSp", output="checkpoints/OpenGprahAU-ResNet50_first_stage.pth"
                    )
                if "ResNet18" in resume:
                    gdown.download(
                        id="1b9yrKF663K9IwY2C2-1SD6azpAdNgBm7", output="checkpoints/OpenGprahAU-ResNet18_first_stage.pth"
                    )

            self.net = load_state_dict(self.net, resume)

        self.net.eval()
        self.net.to(device)

    def face_detection(self, image, expand=True):
        bbox = self.face_detector.run(image)
        if expand:
            bbox = self.expand(image, bbox)
        return bbox

    def _process_image(self, image):
        image = cv2.imread(image) if type(image) == "str" else image

        bbox = self.face_detection(image)
        image_cropped = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        image_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)
        image_cropped = Image.fromarray(image_cropped)
        image_cropped = self.transform(image_cropped).unsqueeze(0).to(self.device)
        return image_cropped

    def run(self, image):
        image_cropped = self._process_image(image)
        with torch.no_grad():
            pred = self.net(image_cropped)
            pred = pred.squeeze().cpu().numpy()

        infostr_probs, infostr_aus = hybrid_prediction_infolist(pred, 0.5)

        print(infostr_aus)
        print(infostr_probs)

        return infostr_aus, infostr_probs


def main():
    open_graph_au = OpenGraphAU()
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        _, frame = cap.read()
        if not _:
            break
        frame = cv2.flip(frame, 1)
        result = open_graph_au.run(frame)

        data = pd.DataFrame.from_dict(result[1], orient="index", columns=["Intensity"])

        sns.barplot(x=data.index, y="Intensity", data=data).figure.savefig("au_intensity.png")
        cv2.imshow("AU Intensity", cv2.imread("au_intensity.png"))
        cv2.imshow("WEBCAM", frame)
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
