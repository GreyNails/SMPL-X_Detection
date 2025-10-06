import os
import os.path as osp
from glob import glob
import shutil

import mmcv
import cv2
import torch
import numpy as np
from mmcv.runner import get_dist_info
import torch.distributed as dist
from pytorch3d.io import save_obj

from config.config import cfg
from util.preprocessing import load_img, augmentation_keep_size
from util.formatting import DefaultFormatBundle
from detrsmpl.data.datasets.pipelines.transforms import Normalize
from detrsmpl.core.conventions.keypoints_mapping import convert_kps
from detrsmpl.core.visualization.visualize_smpl import render_smpl
from detrsmpl.models.body_models.builder import build_body_model
from detrsmpl.utils.ffmpeg_utils import video_to_images

class INFERENCE_demo(torch.utils.data.Dataset):
    def __init__(self, img_dir=None, out_path=None):
        
        self.output_path = out_path
        self.img_dir = img_dir
        self.is_vid = False
        body_model_cfg = dict(
            type='smplx',
            keypoint_src='smplx',
            num_expression_coeffs=10,
            num_betas=10,
            gender='neutral',
            keypoint_dst='smplx_137',
            model_path='data/body_models/smplx',
            use_pca=False,
            use_face_contour=True,
            batch_size = cfg.batch_size)
        self.body_model = build_body_model(body_model_cfg).to('cuda') #cpu
        
        rank, _ = get_dist_info()
        if self.img_dir.endswith('.mp4'):
            self.is_vid = True
            self.img_name = self.img_dir.split('/')[-1][:-4]
            # self.img_dir = self.img_dir[:-4]
        else:
            self.img_name = self.img_dir.split('/')[-1]
        
        self.output_path = os.path.join(self.output_path, self.img_name)
        os.makedirs(self.output_path, exist_ok=True)
        self.mesh_path = os.path.join(self.output_path, 'mesh')
        os.makedirs(self.mesh_path, exist_ok=True)
        self.tmp_dir = os.path.join(self.output_path, 'temp_img')
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.result_img_dir = os.path.join(self.output_path, 'res_img')

        if not self.is_vid:
            if rank == 0:
                image_files = sorted(glob(self.img_dir + '/*.jpg') + glob(self.img_dir + '/*.png'))
                for i, image_file in enumerate(image_files):
                    new_name = os.path.join(self.tmp_dir, '%06d.png'%i)
                    shutil.copy(image_file, new_name)
            # dist.barrier()
        else:
            if rank == 0:
                video_to_images(self.img_dir, self.tmp_dir)
            # dist.barrier()    #single GPU下会报错
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()  # 只有分布式环境已初始化时才调用
        
        self.img_paths = sorted(glob(self.tmp_dir +'/*',recursive=True))
        
        self.num_person = cfg.num_person if 'num_person' in cfg else 0.1
        self.score_threshold = cfg.threshold if 'threshold' in cfg else 0.1  
        self.format = DefaultFormatBundle()
        self.normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
       
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = load_img(self.img_paths[idx],'BGR')
        self.resolution = img.shape[:2]
        img_whole_bbox = np.array([0, 0, img.shape[1],img.shape[0]])
        img, img2bb_trans, bb2img_trans, _, _ = \
            augmentation_keep_size(img, img_whole_bbox, 'test')

        # cropped_img_shape=img.shape[:2]
        img = (img.astype(np.float32)) 
        
        inputs = {'img': img}
        targets = {
            'body_bbox_center': np.array(img_whole_bbox[None]),
            'body_bbox_size': np.array(img_whole_bbox[None])}
        meta_info = {
            'ori_shape':np.array(self.resolution),
            'img_shape': np.array(img.shape[:2]),
            'img2bb_trans': img2bb_trans,
            'bb2img_trans': bb2img_trans,
            'ann_idx': idx}
        result = {**inputs, **targets, **meta_info}
        
        result = self.normalize(result)
        result = self.format(result)
            
        return result
        
    # def inference(self, outs):
    #     img_paths = self.img_paths
    #     for out in outs:
    #         ann_idx = out['image_idx']
    #         scores = out['scores'].clone().cpu().numpy()
    #         img_shape = out['img_shape'].cpu().numpy()[::-1] # w, h
    #         img = cv2.imread(img_paths[ann_idx]) # h, w
    #         scale = img.shape[1]/img_shape[0]
    #         body_bbox = out['body_bbox'].clone().cpu().numpy()
    #         body_bbox = body_bbox * scale
    #         joint_3d, _ =  convert_kps(out['smpl_kp3d'].clone().cpu().numpy(),src='smplx',dst='smplx', approximate=True)

    #         for i, score in enumerate(scores):
    #             if score < self.score_threshold:
    #                 break
    #             if i>self.num_person:
    #                 break
    #             save_name = img_paths[ann_idx].split('/')[-1]
    #             save_name = save_name.split('.')[0] 
    #             vert = out['smpl_verts'][i] + out['cam_trans'][i][None]
    #             # save mesh
    #             exist_result_path = glob(osp.join(self.mesh_path, save_name + '*'))
    #             if len(exist_result_path) == 0:
    #                 person_idx = 0
    #             else:
    #                 last_person_idx = max([
    #                     int(name.split('personId_')[1].split('.obj')[0])
    #                     for name in exist_result_path
    #                 ])
    #                 person_idx = last_person_idx + 1

    #             save_name += '_personId_' + str(person_idx) + '.obj'
    #             save_obj(osp.join(self.mesh_path, save_name), vert, faces=torch.tensor(self.body_model.faces.astype(np.int32)))
            
    #         if i == 0:
    #             save_name = img_paths[ann_idx].split('/')[-1][:-4]
    #             cv2.imwrite(os.path.join(self.result_img_dir,img_paths[ann_idx].split('/')[-1]), img)
    #         else:
    #             verts = out['smpl_verts'][:i] + out['cam_trans'][:i][:, None] 
    #             img = mmcv.imshow_bboxes(img, body_bbox[:i], show=False, colors='green') 
    #             render_smpl(
    #                 verts=verts[None],
    #                 body_model=self.body_model,
    #                 K= np.array(
    #                     [[5000, 0, img_shape[0]/2],
    #                      [0, 5000, img_shape[1]/2],
    #                      [0, 0, 1]]),
    #                 R=None,
    #                 T=None,
    #                 output_path=os.path.join(self.result_img_dir,img_paths[ann_idx].split('/')[-1]),
    #                 image_array=cv2.resize(img, (img_shape[0],img_shape[1]), cv2.INTER_CUBIC),
    #                 in_ndc=False,
    #                 alpha=1,
    #                 convention='opencv',
    #                 projection='perspective',
    #                 overwrite=True,
    #                 no_grad=True,
    #                 device='cuda', #cpu
    #                 resolution=[img_shape[1],img_shape[0]],
    #                 render_choice='hq' 
    #             )
    #     return None
    def inference(self, outs):
        img_paths = self.img_paths
        for out in outs:
            ann_idx = out['image_idx']
            scores = out['scores'].clone().cpu().numpy()
            img_shape = out['img_shape'].cpu().numpy()[::-1] # w, h
            img = cv2.imread(img_paths[ann_idx]) # h, w
            scale = img.shape[1]/img_shape[0]
            body_bbox = out['body_bbox'].clone().cpu().numpy()
            body_bbox = body_bbox * scale
            joint_3d, _ =  convert_kps(out['smpl_kp3d'].clone().cpu().numpy(),src='smplx',dst='smplx', approximate=True)

            for i, score in enumerate(scores):
                if score < self.score_threshold:
                    break
                if i>self.num_person:
                    break
                save_name = img_paths[ann_idx].split('/')[-1]
                save_name = save_name.split('.')[0] 
                vert = out['smpl_verts'][i] + out['cam_trans'][i][None]
                # save mesh
                exist_result_path = glob(osp.join(self.mesh_path, save_name + '*'))
                if len(exist_result_path) == 0:
                    person_idx = 0
                else:
                    last_person_idx = max([
                        int(name.split('personId_')[1].split('.obj')[0])
                        for name in exist_result_path
                    ])
                    person_idx = last_person_idx + 1

                save_name += '_personId_' + str(person_idx) + '.obj'
                save_obj(osp.join(self.mesh_path, save_name), vert, faces=torch.tensor(self.body_model.faces.astype(np.int32)))
            
            if i == 0:
                save_name = img_paths[ann_idx].split('/')[-1][:-4]
                cv2.imwrite(os.path.join(self.result_img_dir,img_paths[ann_idx].split('/')[-1]), img)
            else:
                verts = out['smpl_verts'][:i] + out['cam_trans'][:i][:, None]
                
                # 创建3通道黑色背景
                background = np.zeros((img_shape[1], img_shape[0], 3), dtype=np.uint8)
                
                # 准备输出文件名
                output_filename = img_paths[ann_idx].split('/')[-1]
                output_filename_temp = output_filename.rsplit('.', 1)[0] + '_temp.png'
                output_filename_final = output_filename.rsplit('.', 1)[0] + '.png'
                
                render_smpl(
                    verts=verts[None],
                    body_model=self.body_model,
                    K= np.array(
                        [[5000, 0, img_shape[0]/2],
                        [0, 5000, img_shape[1]/2],
                        [0, 0, 1]]),
                    R=None,
                    T=None,
                    output_path=os.path.join(self.result_img_dir, output_filename_temp),
                    image_array=background,  # 使用黑色背景
                    in_ndc=False,
                    alpha=0.9,
                    convention='opencv',
                    projection='perspective',
                    overwrite=True,
                    no_grad=True,
                    device='cuda',
                    resolution=[img_shape[1],img_shape[0]],
                    render_choice='hq' 
                )
                
                # 后处理：将黑色背景转为透明
                img_rendered = cv2.imread(os.path.join(self.result_img_dir, output_filename_temp), cv2.IMREAD_COLOR)
                
                # 转换为RGBA格式
                img_rgba = cv2.cvtColor(img_rendered, cv2.COLOR_BGR2BGRA)
                
                # 创建mask：识别黑色背景像素（阈值可以调整，10表示接近黑色）
                mask = np.all(img_rendered < 10, axis=2)
                
                # 将黑色背景的alpha通道设为0（完全透明）
                img_rgba[mask, 3] = 0
                
                # 保存为PNG格式以支持透明度
                cv2.imwrite(os.path.join(self.result_img_dir, output_filename_final), img_rgba)
                
                # 删除临时文件
                if os.path.exists(os.path.join(self.result_img_dir, output_filename_temp)):
                    os.remove(os.path.join(self.result_img_dir, output_filename_temp))
        
        return None
