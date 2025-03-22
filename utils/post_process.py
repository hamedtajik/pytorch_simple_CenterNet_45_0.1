
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import _gather_feature, _tranpose_and_gather_feature, flip_tensor


def _nms(heat, kernel=3):
  hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
  keep = (hmax == heat).float()
  return heat * keep


def _topk(scores, K=40):
  batch, cat, height, width = scores.size()

  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  topk_inds = topk_inds % (height * width)
  topk_ys = (topk_inds / width).int().float()
  topk_xs = (topk_inds % width).int().float()

  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  topk_clses = (topk_ind / K).int()
  topk_inds = _gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(hmap, regs, w_h_, K=100):
  batch, cat, height, width = hmap.shape
  hmap=torch.sigmoid(hmap)

  # if flip test
  if batch > 1:
    hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
    w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
    regs = regs[0:1]

  batch = 1

  hmap = _nms(hmap)  # perform nms on heatmaps

  scores, inds, clses, ys, xs = _topk(hmap, K=K)

  regs = _tranpose_and_gather_feature(regs, inds)
  regs = regs.view(batch, K, 2)
  xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
  ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

  w_h_ = _tranpose_and_gather_feature(w_h_, inds)
  w_h_ = w_h_.view(batch, K, 2)

  clses = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)
  bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                      ys - w_h_[..., 1:2] / 2,
                      xs + w_h_[..., 0:1] / 2,
                      ys + w_h_[..., 1:2] / 2], dim=2)
  detections = torch.cat([bboxes, scores, clses], dim=2)
  return detections


def circledet_decode(hmap, regs, r, K=100):
  # print("heatmaps    ", hmap.shape)
  # print("regs    ", regs.shape)
  # print("r    ", r.shape)
  batch, cat, height, width = hmap.shape
  hmap=torch.sigmoid(hmap)

  # if flip test
  if batch > 1:
    hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
    # w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
    r = (r[0:1] + flip_tensor(r[1:2])) / 2
    regs = regs[0:1]

  batch = 1
  print("Before NMS")
  hmap = _nms(hmap)  # perform nms on heatmaps
  print(f"After NMS, hmap shape: {hmap.shape}")
  scores, inds, clses, ys, xs = _topk(hmap, K=K)
  print(f"Top-K detections (K={K}):")
  print(f"Scores shape: {scores.shape}, Indices shape: {inds.shape}, Classes shape: {clses.shape}")

  regs = _tranpose_and_gather_feature(regs, inds)
  regs = regs.view(batch, K, 2)
  xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
  ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

  # w_h_ = _tranpose_and_gather_feature(w_h_, inds)
  # w_h_ = w_h_.view(batch, K, 2)
  r = _tranpose_and_gather_feature(r, inds)
  # Optionally print the regression (offset) and radius values
  print(f"Regs shape: {regs.shape}, Radius shape: {r.shape}")
  r = r.view(batch, K, 1)

  clses = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)
  # bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
  #                     ys - w_h_[..., 1:2] / 2,
  #                     xs + w_h_[..., 0:1] / 2,
  #                     ys + w_h_[..., 1:2] / 2], dim=2)
  # detections = torch.cat([bboxes, scores, clses], dim=2)
  circles = torch.cat([xs,
                      ys,
                      r], dim=2)
  detections = torch.cat([circles, scores, clses], dim=2)
  print(f"Final detections shape: {detections.shape}")
  return detections


