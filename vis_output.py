import pickle as pkl
import json
import torch
import numpy as np
from manopth.manolayer import ManoLayer
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
from PIL import Image
import cv2
from myutils import geom_utils, io_utils, image_utils, camera_utils, vis_utils

with open('/remote-home/jiangshijian/nerf/data/VISOR/GroundTruth-SparseAnnotations/mocap_res/train/P08/P08_09/mocap/P08_09_frame_0000005301_prediction_result.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1')
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
'''
似乎输出的pred_vertices_smpl和ManoLayer输出的顶点不是对应的关系，即两者的face不一样
'''
hand_model = ManoLayer(mano_root='../mano/models', flat_hand_mean=False, use_pca=False)
hand_faces = hand_model.th_faces.numpy()
right_hand = data['pred_output_list'][0]['right_hand']
left_hand = data['pred_output_list'][0]['left_hand']


# pose = torch.from_numpy(right_hand['pred_hand_pose'])
# hand_boxScale_o2n = right_hand['bbox_scale_ratio']
# hand_bboxTopLeft = right_hand['bbox_top_left']
# global_rot, hand_pose = pose[:, :3], pose[:, 3:]
# hand_beta = torch.from_numpy(right_hand['pred_hand_betas'])
# verts, joints = hand_model(pose, hand_beta)
# verts /= 1000.
# joints /= 1000.
# # frankmocap set the joints[5] as the root
# origin_root = joints[:, 5:6].clone()
# cam = right_hand['pred_camera']
# if False:
#     K, pred_full_t = camera_utils.convert_weak_perspective_to_full(cam, 5000, 224, hand_boxScale_o2n, hand_bboxTopLeft)
#     verts_np = verts[0].numpy()
#     verts_np += pred_full_t
#     uv = verts_np @ K.T
#     uv[:, :2] /= uv[:, -1:] # nearly the same as the vert_bboxcoord
#     from renderer.screen_free_visualizer import Visualizer
#     visualizer = Visualizer('pytorch3d')
#     hand_bbox_list = data['hand_bbox_list']
#     pred_mesh_list = [{'vertices': uv, 'faces': hand_model.th_faces.numpy()}]
#     img_original_bgr = cv2.imread(data['image_path'])
#     res_img = visualizer.visualize(
#             img_original_bgr, 
#             pred_mesh_list = pred_mesh_list, 
#             hand_bbox_list = hand_bbox_list)
#     cv2.imwrite('tmp.jpg', res_img)
# K, pred_full_t = camera_utils.convert_weak_perspective_to_full(cam, 5000, 224, hand_boxScale_o2n, hand_bboxTopLeft)
# pred_full_t = torch.from_numpy(pred_full_t).unsqueeze(0)
# global_rot_matrix = geom_utils.axis_angle_to_matrix(global_rot)
# hand_pose = torch.cat([torch.zeros_like(global_rot), hand_pose], dim=1) # For 48-dim, set the global rot to zero
# verts, joints = hand_model(hand_pose, hand_beta)
# verts /= 1000. # transform to the meter
# joints /= 1000.
# wrist_root = joints[:, 0:1].clone()
# # root = joints[:, 0:1] # set the wrist as the root
# verts -= wrist_root
# # get the same results as frankmocap results
# verts = verts @ global_rot_matrix.transpose(1, 2) + wrist_root - origin_root + pred_full_t
# rot_w2c = global_rot_matrix
# trans_w2c = wrist_root - origin_root + pred_full_t
input_img = Image.open(data['image_path'])
vis = vis_utils.Visualizer(img_size=(input_img.size[0], input_img.size[1]))
# render_img = vis.draw_mesh(np.array(input_img) / 255., verts[0].numpy(), hand_model.th_faces.numpy(), K)
# cv2.imwrite('./tmp.jpg', render_img[:, :, ::-1]*255)
'''
transform between the normalized coordinate and the frankmocap output coordinate
'''

if len(right_hand.keys()) != 0:
    vertices = right_hand['pred_vertices_smpl']
    tmp_joints = right_hand['pred_joints_smpl']
    tmp_wrist_root = tmp_joints[0:1].copy()
    vertices -= tmp_wrist_root
    pred_hand_root = right_hand['pred_hand_root']
    pred_global_rot = right_hand['pred_global_rot']
    tmp_pred_full_t = right_hand['pred_full_t']
    tmp_K = right_hand['K']
    # transform to the camera coordinate
    vertices = vertices @ pred_global_rot.T + tmp_wrist_root - pred_hand_root + tmp_pred_full_t
    render_img = vis.draw_mesh(np.array(input_img) / 255., vertices, hand_model.th_faces.numpy(), tmp_K)
else:
    render_img = np.array(input_img) / 255.
if len(left_hand.keys()) != 0:
    vertices = left_hand['pred_vertices_smpl']
    tmp_joints = left_hand['pred_joints_smpl']
    tmp_wrist_root = tmp_joints[0:1].copy()
    vertices -= tmp_wrist_root
    pred_hand_root = left_hand['pred_hand_root']
    pred_global_rot = left_hand['pred_global_rot']
    tmp_pred_full_t = left_hand['pred_full_t']
    tmp_K = left_hand['K']
    vertices = vertices @ pred_global_rot.T + tmp_wrist_root - pred_hand_root + tmp_pred_full_t
    render_img = vis.draw_mesh(render_img, vertices, hand_model.th_faces.numpy(), tmp_K)
cv2.imwrite('./tmp.jpg', render_img[:, :, ::-1]*255)
