from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
# import datasets.eval_protocals.kidpath_circle as kidpath_circle
from . import kidpath_circle as kidpath_circle
# from datasets.eval_protocals.circle_eval import CIRCLEeval
from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import os

import torch.utils.data as data
import cv2

from utils.image import get_border, get_affine_transform, affine_transform, color_aug
from utils.image import draw_umich_gaussian, gaussian_radius

import torch
import math


COCO_MEAN = [0.40789654, 0.44719302, 0.47026115]
COCO_STD = [0.28863828, 0.27408164, 0.27809835]
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]

class NeonTree(data.Dataset):
    num_classes = 1
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, data_dir, split, split_ratio=1.0, img_size=512):
        super(NeonTree, self).__init__()
        ##############
        self.class_name = ['tree']
        self.valid_ids = [1]
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}
        ##############
        self.data_dir = os.path.join(data_dir, 'neontree/RGB')
        # self.img_dir = os.path.join(self.data_dir, '{}'.format(split))
        self.img_dir = self.data_dir

        self.annot_path = os.path.join(
            self.data_dir, 'annotations_small_{}.json').format(split)
            # self.data_dir, 'annotations_{}.json').format(split)

        # self.max_objs = 1000
        ###########################
        self.padding = 31 #for resnet/resdcn
        self.down_ratio = 4
        self.img_size = {'h': img_size, 'w': img_size}
        self.fmap_size = {'h': img_size // self.down_ratio, 'w': img_size // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.7
        self.data_rng = np.random.RandomState(123)
        self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
        self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]
        ###########################
        self.max_objs = 4000
        self.class_name = [
            '__background__', 'nuclei']
        self._valid_ids = [1]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split

        print('==> initializing Neontree {} data.'.format(split))
        print("annot path    ", self.annot_path)
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        # with open(self.annot_path, 'r') as f:
        #     self.images = json.load(f)['images']
        
        self.num_samples = len(self.images)

        self.circle = kidpath_circle.CIRCLE(self.annot_path)
        self.images_circle = self.circle.getImgIds()
        self.num_samples_circle = len(self.images_circle)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def __getitem__(self, index):
      img_id = self.images[index]
      img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
      ann_ids = self.coco.getAnnIds(imgIds=[img_id])
      annotations = self.coco.loadAnns(ids=ann_ids)
      labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations])
      bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32)
      if len(bboxes) == 0:
        bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
        labels = np.array([[0]])
      bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy

      img = cv2.imread(img_path)
      height, width = img.shape[0], img.shape[1]
      center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image
      scale = max(height, width) * 1.0

      flipped = False
      if self.split == 'train':
        scale = scale * np.random.choice(self.rand_scales)
        w_border = get_border(128, width)
        h_border = get_border(128, height)
        center[0] = np.random.randint(low=w_border, high=width - w_border)
        center[1] = np.random.randint(low=h_border, high=height - h_border)

        if np.random.random() < 0.5:
          flipped = True
          img = img[:, ::-1, :]
          center[0] = width - center[0] - 1

      trans_img = get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']])
      img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h']))

      # -----------------------------------debug---------------------------------
      # for bbox, label in zip(bboxes, labels):
      #   if flipped:
      #     bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      #   bbox[:2] = affine_transform(bbox[:2], trans_img)
      #   bbox[2:] = affine_transform(bbox[2:], trans_img)
      #   bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.img_size['w'] - 1)
      #   bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.img_size['h'] - 1)
      #   cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
      #   cv2.putText(img, self.class_name[label + 1], (int(bbox[0]), int(bbox[1])),
      #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
      # cv2.imshow('img', img)
      # cv2.waitKey()
      # -----------------------------------debug---------------------------------

      img = img.astype(np.float32) / 255.

      if self.split == 'train':
        color_aug(self.data_rng, img, self.eig_val, self.eig_vec)

      img -= self.mean
      img /= self.std
      img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

      trans_fmap = get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']])

      hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
      # w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
      r = np.zeros((self.max_objs, 1), dtype=np.float32)  # width and height
      regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
      inds = np.zeros((self.max_objs,), dtype=np.int64)
      ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

      # detections = []
    #   skipped_count = 0
      for k, (bbox, label) in enumerate(zip(bboxes, labels)):
        if flipped:
          bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        bbox[:2] = affine_transform(bbox[:2], trans_fmap)
        bbox[2:] = affine_transform(bbox[2:], trans_fmap)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h > 0 and w > 0:
          obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
          obj_c_int = obj_c.astype(np.int32)

          radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
          draw_umich_gaussian(hmap[label], obj_c_int, radius)
          # w_h_[k] = 1. * w, 1. * h
          
          if k < self.max_objs:
            r[k] = np.min([(h)/2, (w)/2])
          else:
            #    print(f"Warning: More than {self.max_objs} objects in the image. Skipping this object.")
            #    print(f"Image with {k } objects is being skipped because it exceeds the limit.")
               continue
          
        #   print(f"Object c values: obj_c={obj_c}, obj_c_int={obj_c_int}")
        #   print(f"Current index k={k}")
          regs[k] = obj_c - obj_c_int  # discretization error
          inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
          ind_masks[k] = 1
          # groundtruth bounding box coordinate with class
          # detections.append([obj_c[0] - w / 2, obj_c[1] - h / 2,
          #                    obj_c[0] + w / 2, obj_c[1] + h / 2, 1, label])

      # detections = np.array(detections, dtype=np.float32) \
      #   if len(detections) > 0 else np.zeros((1, 6), dtype=np.float32)

    #   print(f"Image skipped, total skipped so far: {skipped_count}")
      return {'image': img,
              # 'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
              'hmap': hmap, 'radius': r, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
              'c': center, 's': scale, 'img_id': img_id}

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def convert_eval_circle_format(self, all_circles):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_circles:
            for cls_ind in all_circles[image_id]:
                try:
                    category_id = self._valid_ids[cls_ind - 1]
                except:
                    aaa  =1
                for circle in all_circles[image_id][cls_ind]:
                    score = circle[3]
                    circle_out = list(map(self._to_float, circle[0:3]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "score": float("{:.2f}".format(score)),
                        'circle_center': [circle_out[0], circle_out[1]],
                        'circle_radius': circle_out[2]
                    }
                    if len(circle) > 5:
                        extreme_points = list(map(self._to_float, circle[5:13]))
                        detection["extreme_points"] = extreme_points

                    # output_h = 512  # hard coded
                    # output_w = 512  # hard coded
                    # cp = [0, 0]
                    # cp[0] = circle_out[0]
                    # cp[1] = circle_out[1]
                    # cr = circle_out[2]
                    # if cp[0] - cr < 0 or cp[0] + cr > output_w:
                    #     continue
                    # if cp[1] - cr < 0 or cp[1] + cr > output_h:
                    #     continue

                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        print(" - save dir   ", save_dir)
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        # coco_dets = results
        # print(results)
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


    def save_circle_results(self, results, save_dir):
        json.dump(self.convert_eval_circle_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_circle_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_circle_results(results, save_dir)
        circle_dets = self.circle.loadRes('{}/results.json'.format(save_dir))
        # gt = kidpath_circle.CIRCLE('{}/circle_annotations.json'.format(save_dir))
        # with open('{}/circle_annotations.json'.format(save_dir),'r') as f:
        #     gt_dets = json.load(f)
        # print("dets    ", circle_dets)
        # print("self.circle   ", self.circle)
        # llll += 3
        circle_eval = CIRCLEeval(self.circle, circle_dets, "circle")
        # circle_eval = CIRCLEeval(gt_dets, circle_dets, "circle")
        # circle_eval = CIRCLEeval(self.circle, circle_dets, "circle_box")
        circle_eval.evaluate()
        circle_eval.accumulate()
        circle_eval.summarize()




class Neon_eval(NeonTree):
  def __init__(self, data_dir, split, test_scales=(1,), test_flip=False, fix_size=False):
    super(Neon_eval, self).__init__(data_dir, split)
    self.test_flip = test_flip
    self.test_scales = test_scales
    self.fix_size = fix_size

  def __getitem__(self, index):
    img_id = self.images[index]
    img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
    print("image path    ", img_path)
    image = cv2.imread(img_path)
    height, width = image.shape[0:2]

    out = {}
    for scale in self.test_scales:
      new_height = int(height * scale)
      new_width = int(width * scale)

      if self.fix_size:
        img_height, img_width = self.img_size['h'], self.img_size['w']
        center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        scaled_size = max(height, width) * 1.0
        scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)
      else:
        img_height = (new_height | self.padding) + 1
        img_width = (new_width | self.padding) + 1
        center = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        scaled_size = np.array([img_width, img_height], dtype=np.float32)

      img = cv2.resize(image, (new_width, new_height))
      trans_img = get_affine_transform(center, scaled_size, 0, [img_width, img_height])
      img = cv2.warpAffine(img, trans_img, (img_width, img_height))

      img = img.astype(np.float32) / 255.
      img -= self.mean
      img /= self.std
      img = img.transpose(2, 0, 1)[None, :, :, :]  # from [H, W, C] to [1, C, H, W]

      if self.test_flip:
        img = np.concatenate((img, img[:, :, :, ::-1].copy()), axis=0)

      out[scale] = {'image': img,
                    'center': center,
                    'scale': scaled_size,
                    'fmap_h': img_height // self.down_ratio,
                    'fmap_w': img_width // self.down_ratio}

    return img_id, out

  def convert_eval_format(self, all_bboxes):
    # all_bboxes: num_samples x num_classes x 5
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self.valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out = list(map(lambda x: float("{:.2f}".format(x)), bbox[0:4]))

          detection = {"image_id": int(image_id),
                       "category_id": int(category_id),
                       "bbox": bbox_out,
                       "score": float("{:.2f}".format(score))}
          detections.append(detection)
    return detections
  

  def convert_eval_circle_format(self, all_circles):
    # import pdb; pdb.set_trace()
    detections = []
    i = 0
    for image_id in all_circles:
        for cls_ind in all_circles[image_id]:
            try:
                category_id = self._valid_ids[cls_ind - 1]
            except:
                aaa  =1
            for circle in all_circles[image_id][cls_ind]:
                score = circle[3]
                circle_out = list(map(self._to_float, circle[0:3]))

                detection = {
                    "id": int(i),
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "score": float("{:.2f}".format(score)),
                    'circle_center': [circle_out[0], circle_out[1]],
                    'circle_radius': circle_out[2],
                    'area': np.pi * circle_out[2]**2
                }
                if len(circle) > 5:
                    extreme_points = list(map(self._to_float, circle[5:13]))
                    detection["extreme_points"] = extreme_points
                
                i += 1
                detections.append(detection)
    return detections

  def save_circle_results(self, results, save_dir):
    print("-----------------     save circle")
    # json.dump(self.convert_eval_circle_format(results),
    json.dump(results,
              open('{}/results.json'.format(save_dir), 'w'))

  def run_eval(self, results, save_dir=None):
    # detections = self.convert_eval_format(results)
    detections = self.convert_eval_circle_format(results)
    print("detections   ", detections[0])
    print(" -- save dir    ", save_dir)
    # if save_dir is not None:
    #   result_json = os.path.join(save_dir, "results.json")
    #   json.dump(detections, open(result_json, "w"))
    # self.save_circle_results(results, save_dir)
    self.save_circle_results(detections, save_dir)

    # coco_dets = self.coco.loadRes(detections)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    # coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval = CIRCLEeval(self.coco, coco_dets, 'circle')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

  @staticmethod
  def collate_fn(batch):
    out = []
    for img_id, sample in batch:
      out.append((img_id, {s: {k: torch.from_numpy(sample[s][k]).float()
      if k == 'image' else np.array(sample[s][k]) for k in sample[s]} for s in sample}))
    return out




import numpy as np
import datetime
import time
from collections import defaultdict
from . import mask as maskUtils
import copy

class CIRCLEeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                print("ann    ", ann)
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            # gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            # dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=[]))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=[]))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            # print("gts   ", gt['image_id'])
            self._gts[gt['image_id'], gt['category_id']].append(gt)
            # self._gts[gt['image_id'], 1].append(gt)
        for dt in dts:
            # print("dts   ", dt['image_id'])
            self._dts[dt['image_id'], dt['category_id']].append(dt)
            # self._dts[dt['image_id'], 1].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        # jjjjjj += 4
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'circle':
            computeIoU = self.computeCircleIoU
        elif p.iouType == 'circle_box':
            computeIoU = self.computeCircleBoxIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious
    
    def convert_region_seg_to_circle(self, data): ## my ydit
        # For each ROI
        anns_all = data['annotations']
        for i, ann in enumerate(anns_all):
            vertices_seg = ann['segmentation'][0]

            assert len(vertices_seg) % 2 == 0
            cnt_seg = np.zeros((len(vertices_seg) // 2, 2))
            # For each vertex
            # Note: COCO encodes polygons as [x1, y1, ..., xn, yn], so divide index by two.
            for vi in range(len(vertices_seg)):
                if vi % 2 == 0:
                    cnt_seg[vi//2, 0] = vertices_seg[vi]
                else:
                    cnt_seg[vi//2, 1] = vertices_seg[vi]

            x,y,r = numerical_stable_circle(cnt_seg)

            anns_all[i]['circle_center'] = [x, y]
            anns_all[i]['circle_radius'] = r
        data['annotations'] = anns_all
        return data

    def computeCircleIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            # print(" - 1  ")
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            # print(" - 2  ")
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        # if len(dt) > p.maxDets[-1]:
        #     dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        elif p.iouType == 'circle':
            # print("gt    ", gt[0:3])
            # print("\ndt    ", dt[0:3])
            print("--------------------------------------")
            # print("gt    ", len(gt))
            # if len(gt) == 0:
            #     print("gttt   ", gt)
            # print("dt    ", len(dt))
            g = [g['circle_center']+[g['circle_radius']] for g in gt]
            d = [d['circle_center']+[d['circle_radius']]for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        # iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.circleIOU(d,g)
        # print("ious    ", ious)
        return ious

    def computeCircleBoxIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        elif p.iouType == 'circle_box':
            g = [g['circle_center']+[g['circle_radius']] for g in gt]
            d = [d['circle_center']+[d['circle_radius']]for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.circleBoxIOU(d,g)
        return ious


    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            # print("gt    ", g)
            # print("arng    ", aRng)
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=1000):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            ######
            with open("results_test.csv", "a") as myfile:
                myfile.write(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)+'\n')
            ######
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            with open("results_test.csv", "a") as myfile:
                myfile.write('---------\n')
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'circle' or iouType == 'circle_box':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1000, 1000, 1000]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1000]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'circle' or iouType == 'circle_box':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
