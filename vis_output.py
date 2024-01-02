import os
import pickle as pkl
import json
import torch
import numpy as np
from manopth.manolayer import ManoLayer
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
from PIL import Image
import cv2
from glob import glob
from tqdm import tqdm
from myutils import geom_utils, io_utils, image_utils, camera_utils, vis_utils

pkl_file_list = glob('/remote-home/jiangshijian/nerf/hoi_tools/kettle/mocap_res/mocap/*.pkl')
# demo_type:<class 'str'>
# smpl_type:<class 'str'>
# image_path:<class 'str'>
# body_bbox_list:<class 'list'>
# hand_bbox_list:<class 'list'>
# save_mesh:<class 'bool'>
# pred_output_list:<class 'list'>
# data['pred_output_list'][0]['right_hand'].keys()
# dict_keys(['pred_vertices_smpl', 'pred_joints_smpl', 'faces', 'bbox_scale_ratio', 'bbox_top_left', 'pred_camera', 'img_cropped', 'pred_hand_pose', 'pred_hand_betas', 'pred_vertices_img', 'pred_joints_img'])
# pred_hand_pose: 1, 48
# pred_hand_betas: 1, 10
# pred_camera: 3,
for pkl_file in tqdm(pkl_file_list):
    with open(pkl_file, 'rb') as f:
        data = pkl.load(f, encoding='latin1')
    '''
    似乎输出的pred_vertices_smpl和ManoLayer输出的顶点不是对应的关系，即两者的face不一样
    '''
    hand_model = ManoLayer(mano_root='../mano/models', flat_hand_mean=False, use_pca=False)
    hand_faces = hand_model.th_faces.numpy()
    hand_model_left = ManoLayer(mano_root='../mano/models', side='left', flat_hand_mean=False, use_pca=False)
    hand_faces_left = hand_model_left.th_faces.numpy()
    right_hand = data['pred_output_list'][0]['right_hand']
    left_hand = data['pred_output_list'][0]['left_hand']

    input_img = Image.open(data['image_path']).convert('RGB')
    vis = vis_utils.Visualizer(img_size=(input_img.size[0], input_img.size[1]))
    '''
    transform between the normalized coordinate and the frankmocap output coordinate
    '''

    if len(right_hand.keys()) != 0:
        vertices = right_hand['pred_vertices_smpl']
        tmp_joints = right_hand['pred_joints_smpl']
        tmp_wrist_root = tmp_joints[0:1].copy()
        # vertices -= tmp_wrist_root
        pred_hand_root = right_hand['pred_hand_root']
        pred_global_rot = right_hand['pred_global_rot']
        tmp_pred_full_t = right_hand['pred_full_t']
        tmp_K = right_hand['K']
        # transform to the camera coordinate w2c_rot: pred_global_rot w2c_trans: tmp_wrist_root - pred_hand_root + tmp_pred_full_t
        vertices = vertices @ pred_global_rot.T + tmp_wrist_root - pred_hand_root + tmp_pred_full_t - tmp_wrist_root @ pred_global_rot.T
        render_img = vis.draw_mesh(np.array(input_img) / 255., vertices, hand_faces, tmp_K)
        right_bbox = data['hand_bbox_list'][0]['right_hand']
        right_bbox[2:] += right_bbox[0:2]
        bbox_img = vis.draw_bbox(input_img, right_bbox, (0, 255, 0))
    else:
        render_img = np.array(input_img) / 255.
        bbox_img = np.array(input_img)
    if len(left_hand.keys()) != 0:
        vertices = left_hand['pred_vertices_smpl']
        tmp_joints = left_hand['pred_joints_smpl']
        tmp_wrist_root = tmp_joints[0:1].copy()
        # vertices -= tmp_wrist_root
        pred_hand_root = left_hand['pred_hand_root']
        pred_global_rot = left_hand['pred_global_rot']
        tmp_pred_full_t = left_hand['pred_full_t']
        tmp_K = left_hand['K']
        vertices = vertices @ pred_global_rot.T + tmp_wrist_root - pred_hand_root + tmp_pred_full_t - tmp_wrist_root @ pred_global_rot.T
        render_img = vis.draw_mesh(render_img, vertices, hand_faces_left, tmp_K)
        left_bbox = data['hand_bbox_list'][0]['left_hand']
        left_bbox[2:] += left_bbox[0:2]
        bbox_img = vis.draw_bbox(Image.fromarray(bbox_img), left_bbox, (0, 0, 255))
    cv2.imwrite(os.path.join('/remote-home/jiangshijian/nerf/hoi_tools/kettle/mocap_res/rendered','./{}.jpg'.format(pkl_file.split('/')[-1][:-4])), np.hstack([bbox_img / 255., render_img])[:, :, ::-1]*255)
