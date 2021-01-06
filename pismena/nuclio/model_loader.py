from fastai.vision import *
from helper.object_detection_helper import *
from loss.RetinaNetFocalLoss import RetinaNetFocalLoss
from models.RetinaNet import RetinaNet
import locale

locale.setlocale(locale.LC_ALL, 'C')
import numpy as np
import torch
import torchvision.models as models
import torchvision

class ModelLoader:
    def __init__(self):
        anchor_sizes = [(32, 32), (16, 16)]
        anchor_ratios = [1.4, 1.2, 1, 0.8, 0.6]
        anchor_scales = [0.3, 0.4, 0.6, 0.8, 1, 1.2]
        n_anch = len(anchor_ratios) * len(anchor_scales)
        anchors = create_anchors(sizes=anchor_sizes, ratios=anchor_ratios, scales=anchor_scales)

        n_classes = 2
        crit = RetinaNetFocalLoss(anchors)
        encoder = create_body(models.resnet18, True, -2)
        self.model = RetinaNet(encoder, n_classes=n_classes, n_anchors=n_anch, sizes=[32, 16], chs=32, final_bias=-4.,
                          n_conv=2)
        state_dic = torch.load('./round2_pretraining2.pth', map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dic['model'])
        self.model.eval()

    def infer(self, image, context, detect_thresh: float = 0.2, nms_thresh: float = 0.3):
        anchor_sizes = [(32, 32), (16, 16)]
        anchor_ratios = [1.4, 1.2, 1, 0.8, 0.6]
        anchor_scales = [0.3, 0.4, 0.6, 0.8, 1, 1.2]
        n_anch = len(anchor_ratios) * len(anchor_scales)
        anchors = create_anchors(sizes=anchor_sizes, ratios=anchor_ratios, scales=anchor_scales)

        with torch.no_grad():
            context.logger.info(f"Input image: {image.size()}")
            in_img = torch.load('./tensor.pt')
            context.logger.info(f"in_img size: {in_img.size()}")
            context.logger.info(f"in_img: {in_img}")

            context.logger.info(f"image size: {image.size()}")
            context.logger.info(f"image: {image}")

            test_prediction = self.model(image)
            class_pred_im, bbox_pred_im = test_prediction[:2]
            context.logger.info(f"Prediction output: {len(bbox_pred_im)}")

            context.logger.info(f"Starting  process_output2")
            bbox_pred, scores, preds, detect_count = self.process_output2(
                clas_pred=class_pred_im,
                bbox_pred=bbox_pred_im,
                anchors=anchors)
            context.logger.info(f"detect_count: {len(detect_count)}")
            bbox_preds = []
            start = 0
            for i in detect_count:
                bbox_pred_i = bbox_pred[start:start + i]
                scores_i = scores[start:start + i]
                preds_i = preds[start:start + i]
                bbox_preds.append(self.filter_slices(bbox_pred_i, scores_i, preds_i, nms_thresh, context))
                start += i
            context.logger.info(f"return bbox_preds: {len(bbox_preds)}")
            return bbox_preds

    def process_output2(self, clas_pred, bbox_pred, anchors):
        detect_thresh = 0.25
        bbox_pred = activ_to_bbox(bbox_pred, anchors)
        clas_pred = torch.sigmoid(clas_pred)
        detect_mask = clas_pred.max(2)[0] > detect_thresh
        detect_count = torch.sum(detect_mask, 1)

        if np.array(detect_mask).max() == 0:
            return None, None, None

        bbox_pred, clas_pred = bbox_pred[detect_mask], clas_pred[detect_mask]

        bbox_pred = tlbr2cthw(torch.clamp(cthw2tlbr(bbox_pred), min=-1, max=1))

        scores, preds = clas_pred.max(1)

        return bbox_pred, scores, preds, detect_count

    def filter_slices(self, bbox_pred, scores, preds, nms_thresh, context):
        if bbox_pred is not None:
            to_keep = nms(bbox_pred, scores, context=context, thresh=nms_thresh)
            #to_keep = torchvision.ops.nms(bbox_pred, scores, nms_thresh)

            bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[to_keep].cpu()
            t_sz = torch.Tensor([256, 256])[None].cpu()
        if bbox_pred is not None:
            bbox_pred = to_np(rescale_boxes(bbox_pred, t_sz))
            # change from center to top left
            bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2

        return bbox_pred
