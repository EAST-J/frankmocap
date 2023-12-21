import pickle as pkl
import json
import torch
from manopth.manolayer import ManoLayer
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
import cv2
from myutils import geom_utils, io_utils, image_utils, camera_utils

with open('/remote-home/jiangshijian/nerf/hoi_tools/frankmocap/tmp/mocap/00000_prediction_result.pkl', 'rb') as f:
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
hand_model = ManoLayer(mano_root='../mano/models', flat_hand_mean=False, use_pca=False)
right_hand = data['pred_output_list'][0]['right_hand']
pose = torch.from_numpy(right_hand['pred_hand_pose'])
hand_boxScale_o2n = right_hand['bbox_scale_ratio']
hand_bboxTopLeft = right_hand['bbox_top_left']
global_rot, hand_pose = pose[:, :3], pose[:, 3:]
hand_beta = torch.from_numpy(right_hand['pred_hand_betas'])
verts, joints = hand_model(pose, hand_beta)
verts /= 1000.
joints /= 1000.
# frankmocap set the joints[5] as the root
verts -= joints[:, 5:6, :]
cam = right_hand['pred_camera']
vert_bboxcoord = convert_smpl_to_bbox(
    verts[0].numpy(), cam[0], cam[1:], bAppTransFirst=True)
vert_imgcoord = convert_bbox_to_oriIm(
                                vert_bboxcoord, hand_boxScale_o2n, hand_bboxTopLeft, 
                                512, 512)
if True:
    K, pred_full_t = camera_utils.convert_weak_perspective_to_full(cam, 5000, 224, hand_boxScale_o2n, hand_bboxTopLeft)
    verts_np = verts[0].numpy()
    verts_np += pred_full_t
    uv = verts_np @ K.T
    uv[:, :2] /= uv[:, -1:] # nearly the same as the vert_bboxcoord
    from renderer.screen_free_visualizer import Visualizer
    visualizer = Visualizer('pytorch3d')
    hand_bbox_list = data['hand_bbox_list']
    pred_mesh_list = [{'vertices': vert_imgcoord, 'faces': hand_model.th_faces.numpy()}]
    img_original_bgr = cv2.imread(data['image_path'])
    res_img = visualizer.visualize(
            img_original_bgr, 
            pred_mesh_list = pred_mesh_list, 
            hand_bbox_list = hand_bbox_list)
    cv2.imwrite('tmp.jpg', res_img)

# global_rot_matrix = geom_utils.axis_angle_to_matrix(global_rot)
# hand_pose = torch.cat([torch.zeros_like(global_rot), hand_pose], dim=1) # For 48-dim, set the global rot to zero
# verts, joints = hand_model(hand_pose, hand_beta)
# verts /= 1000. # transform to the meter
# joints /= 1000.
# root = joints[:, 0:1] # set the wrist as the root
# verts -= root
'''
transform between the normalized coordinate and the frankmocap output coordinate
'''
# TODO: 将坐标系放到Wrist的原点，计算R，T变换的矩阵

