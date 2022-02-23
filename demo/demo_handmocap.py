# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
import json
import torch
from torchvision.transforms import Normalize, ToTensor, Resize

from demo.demo_options import DemoOptions
from hand_utils import ManopthWrapper, cvt_axisang_t_i2o, cvt_axisang_t_o2i
from handmocap.hand_modules.h3dw_model import extract_hand_output
import mocap_utils.general_utils as gnu
import mocap_utils.demo_utils as demo_utils

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import time
from jutils import geom_utils, image_utils, mesh_utils
from pytorch3d.renderer import OrthographicCameras, PerspectiveCameras
from pytorch3d.transforms import Translate
from pytorch3d.structures import Meshes


def run_hand_mocap(args, bbox_detector, hand_mocap, visualizer):
    hand_wrapper = ManopthWrapper('/checkpoint/yufeiy2/pretrain_model/smplx/mano').cuda()
    #Set up input data (images or webcam)
    input_type, input_data = demo_utils.setup_input(args)
 
    assert args.out_dir is not None, "Please specify output dir to store the results"
    cur_frame = args.start_frame
    video_frame = 0

    while True:
        # load data
        load_bbox = False

        if input_type =='image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr  = cv2.imread(image_path)
            else:
                img_original_bgr = None

        elif input_type == 'bbox_dir':
            if cur_frame < len(input_data):
                print("Use pre-computed bounding boxes")
                image_path = input_data[cur_frame]['image_path']
                hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                body_bbox_list = input_data[cur_frame]['body_bbox_list']
                img_original_bgr  = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == 'video':      
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        
        elif input_type == 'webcam':
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        cur_frame +=1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break   
        print("--------------------------------------")

        # bbox detection
        if load_bbox:
            body_pose_list = None
            raw_hand_bboxes = None
        elif args.crop_type == 'hand_crop':
            # hand already cropped, thererore, no need for detection
            img_h, img_w = img_original_bgr.shape[:2]
            body_pose_list = None
            raw_hand_bboxes = None
            hand_bbox_list = [ dict(right_hand = np.array([0, 0, img_w, img_h])) ]
        else:            
            # Input images has other body part or hand not cropped.
            # Use hand detection model & body detector for hand detection
            assert args.crop_type == 'no_crop'
            detect_output = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
            body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = detect_output
        
        # save the obtained body & hand bbox to json file
        if args.save_bbox_output:
            demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

        if len(hand_bbox_list) < 1:
            print(f"No hand deteced: {image_path}")
            continue
    
        # Hand Pose Regression
        pred_output_list = hand_mocap.regress(
                img_original_bgr, hand_bbox_list, add_margin=True)
        assert len(hand_bbox_list) == len(body_bbox_list)
        assert len(body_bbox_list) == len(pred_output_list)

        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # visualize
        res_img = visualizer.visualize(
            img_original_bgr, 
            pred_mesh_list = pred_mesh_list, 
            hand_bbox_list = hand_bbox_list)

        # show result in the screen
        if not args.no_display:
            res_img = res_img.astype(np.uint8)
            ImShow(res_img)

        # save the image (we can make an option here)
        if args.out_dir is not None:
            demo_utils.save_res_img(args.out_dir, image_path, res_img)
            # my_vis(img_original_bgr, pred_output_list, hand_wrapper,  osp.join(args.out_dir, 'crop', osp.basename(image_path) + '_my.jpg'))

        # save predictions to pkl
        if args.save_pred_pkl:
            demo_type = 'hand'
            demo_utils.save_pred_to_pkl(
                args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)

        print(f"Processed : {image_path}")
        
    #save images as a video
    if not args.no_video_out and input_type in ['video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    # When everything done, release the capture
    if input_type =='webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()


def my_vis(orig_img, pred_list, hand_wrapper, fname):      
    one_hand = pred_list[0]['right_hand']
    image = one_hand['img_cropped']
    device = 'cuda:0'

    pose = torch.FloatTensor(one_hand['pred_hand_pose']).to(device)
    rot, hA = pose[..., :3], pose[..., 3:]
    hA = hA.clone() + hand_wrapper.hand_mean
    
    # glb = geom_utils.matrix_to_se3(geom_utils.axis_angle_t_to_matrix(rot))
    t = torch.zeros_like(rot)
    # rot, t = cvt_axisang_t_o2i(rot, t)
    glb = geom_utils.matrix_to_se3(geom_utils.axis_angle_t_to_matrix(rot, t))
    wHand, wJoints = hand_wrapper(glb, hA, th_betas=torch.FloatTensor(one_hand['pred_hand_betas']).to(device))
    wHand = wHand.update_padded(wHand.verts_padded() - wJoints[:, 5:6])

    # f = 100
    cam = one_hand['pred_camera']
    new_center = one_hand['bbox_top_left'] + 112 / one_hand['bbox_scale_ratio']  - 100
    new_size = 224 / one_hand['bbox_scale_ratio']  * 2
    resize=300
    image = crop_image(orig_img, new_center, new_size, resize)
    cam, topleft, scale = image_utils.crop_weak_cam(cam, one_hand['bbox_top_left'], one_hand['bbox_scale_ratio'], 
        new_center, new_size, resize=resize)

    def get_cam_fp(cam):
        s, tx, ty = torch.split(torch.FloatTensor(cam), 1, dim=-1)
        fx = 10
        px = 1
        f = torch.FloatTensor([fx, fx])
        p = torch.FloatTensor([px, px])

        # translate = torch.FloatTensor([tx, ty, fx/s]).to(device)
        translate = mesh_utils.weak_to_full_persp(f[0:1], p.unsqueeze(0), s, torch.cat([tx, ty], dim=-1))
        translate = translate.to(device)[0]
        print(cam, one_hand['pred_camera'], translate)
        return translate, f, p

    def get_cTh(hA, rot, translate, ):
        _, joints = hand_wrapper(
            geom_utils.matrix_to_se3(geom_utils.axis_angle_t_to_matrix(rot[None])), 
            hA[None])
        
        cTh = geom_utils.axis_angle_t_to_matrix(
            rot, translate - joints[0, 5])
        return cTh

    translate, f, p, = get_cam_fp(cam)
    cTh = get_cTh(hA[0], rot[0], translate)
    cHand, _ = hand_wrapper(geom_utils.matrix_to_se3(cTh[None]), hA)
    cameras = PerspectiveCameras(f[None], p[None], device='cuda:0')
    # cameras = OrthographicCameras(s, torch.stack([s*tx, s*ty], -1), device=device)
    # cHand = wHand
    print(cTh, translate)

    render = mesh_utils.render_mesh(cHand, cameras, out_size=resize)
    image_list = mesh_utils.render_geom_rot(cHand, scale_geom=True)
    
    image_inp = Resize(resize)(ToTensor()(image).to(device))
    out = render['image'] * render['mask'] + image_inp * (1 - render['mask'])
    image_utils.save_images(out, osp.join(fname + '_hand'))
    image_utils.save_gif(image_list, fname + '_hand')
    cv2.imwrite(fname + '_inp.jpg', image)


def crop_image(img_ori, new_center, new_size, resize):
    x, y = new_center
    h = new_size / 2
    bbox = [x - h, y - h, x + h, y + h]
    image = image_utils.crop_resize(img_ori, bbox, resize)
    return image


    
def main():
    args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    #Set Bbox detector
    bbox_detector =  HandBboxDetector(args.view_type, device)

    # Set Mocap regressor
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)

    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    # run
    run_hand_mocap(args, bbox_detector, hand_mocap, visualizer)
   

if __name__ == '__main__':
    main()
