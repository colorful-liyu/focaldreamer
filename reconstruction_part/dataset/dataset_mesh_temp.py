# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch
from render import util
from render import render
from render import mesh
from render import light

# the camera parameters of video.mp4
def get_camera_params(resolution= 512, fov=45, elev_angle=-20, azim_angle=0):
    fovy   = np.deg2rad(fov) 
    elev = np.radians( elev_angle )
    azim = np.radians( azim_angle ) 
    proj_mtx = util.perspective(fovy, resolution /resolution, 1, 50)
    mv     = util.translate(0, 0, -3) @ (util.rotate_x(elev) @ util.rotate_y(azim))
    normal_rotate =  util.rotate_y_1(-azim ) @ util.rotate_x_1(-elev) 
    # nomral_rotate =  util.rotate_y_1(0) @ util.rotate_x_1(0) 
    mvp    = proj_mtx @ mv
    campos = torch.linalg.inv(mv)[:3, 3]
    bkgs = torch.ones(1, resolution, resolution, 3, dtype=torch.float32, device='cuda')
    return {
        'mvp' : mvp[None, ...].cuda(),
        'mv' : mv[None, ...].cuda(),
        'campos' : campos[None, ...].cuda(),
        'resolution' : [resolution, resolution], 
        'spp' : 1,
        'background' : bkgs,
        'normal_rotate' : normal_rotate[None,...].cuda(),
        }
        
class DatasetMesh(torch.utils.data.Dataset):

    def __init__(self, glctx, FLAGS, validate=False, gif=False):
        # Init 
        self.ref_mesh         = mesh.load_mesh(FLAGS.ref_mesh, FLAGS.mtl_override)
        self.glctx              = glctx
        self.FLAGS              = FLAGS
        self.validate           = validate
        self.gif                = gif
        self.aspect             = FLAGS.train_res[1] / FLAGS.train_res[0]
        self.fovy_range_min     = np.deg2rad(FLAGS.fovy_range[0])
        self.fovy_range_max     = np.deg2rad(FLAGS.fovy_range[1])
        self.elevation_range_min= np.deg2rad(FLAGS.elevation_range[0])
        self.elevation_range_max= np.deg2rad(FLAGS.elevation_range[1])
        self.envlight           = FLAGS.lgt

        #----centering------#
        aabb = mesh.aabb(self.ref_mesh)
        
        self.ref_mesh = mesh.center_by_reference(self.ref_mesh, aabb, FLAGS.base_mesh_scale[0][0])
        
        if self.FLAGS.local_rank == 0:
            print("DatasetMesh: ref mesh has %d triangles and %d vertices" % (self.ref_mesh.t_pos_idx.shape[0], self.ref_mesh.v_pos.shape[0]))

        # Sanity test training texture resolution
        ref_texture_res = np.maximum(self.ref_mesh.material['kd'].getRes(), self.ref_mesh.material['ks'].getRes())
        if 'normal' in self.ref_mesh.material:
            ref_texture_res = np.maximum(ref_texture_res, self.ref_mesh.material['normal'].getRes())
        if self.FLAGS.local_rank == 0 and FLAGS.texture_res[0] < ref_texture_res[0] or FLAGS.texture_res[1] < ref_texture_res[1]:
            print("---> WARNING: Picked a texture resolution lower than the reference mesh [%d, %d] < [%d, %d]" % (FLAGS.texture_res[0], FLAGS.texture_res[1], ref_texture_res[0], ref_texture_res[1]))

        # Load environment map texture
        self.envlight = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)
        
        self.ref_mesh = mesh.compute_tangents(self.ref_mesh)
        
    def _gif_scene(self, itr):
        fovy = np.deg2rad(45)
        proj_mtx = util.perspective(fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        ang    = (itr / 100) * np.pi * 2
        rotate_x =  np.deg2rad(20)
        prompt_index = 0
        mv     = util.translate(0, 0, -3) @ (util.rotate_x(-rotate_x) @ util.rotate_y(ang ))
        normal_rotate =  util.rotate_y_1(0)
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...], mvp[None, ...], campos[None, ...], self.FLAGS.display_res, self.FLAGS.spp, normal_rotate[None,...], prompt_index
    
    def _validate_scene(self, itr):
        fovy = np.deg2rad(45)
        proj_mtx = util.perspective(fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        ang    = (itr / 4) * np.pi * 2
        rotate_x = np.random.uniform(-np.pi/4,np.pi/18)
        prompt_index = 0
        mv     = util.translate(0, 0, -3) @ (util.rotate_x(rotate_x) @ util.rotate_y( ang ))
        normal_rotate =  util.rotate_y_1(-ang)@util.rotate_x_1(-rotate_x) 
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]
        return mv[None, ...], mvp[None, ...], campos[None, ...], self.FLAGS.display_res, self.FLAGS.spp, normal_rotate[None,...], prompt_index

    def _train_scene(self, itr):
        fovy =  np.random.uniform(self.fovy_range_min, self.fovy_range_max)
        proj_mtx = util.perspective(fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        if self.FLAGS.gpu_number == 8: # All the results in the paper were generated using 8 3090 GPUs. We cannot guarantee that fewer than 8 GPUs can achieve the same effect.
            if self.FLAGS.local_rank in [0,4]:
                rotate_y = np.random.uniform(np.deg2rad(-45), np.deg2rad(45))
            elif self.FLAGS.local_rank in [1,5]:
                rotate_y = np.random.uniform(np.deg2rad(45), np.deg2rad(135))
            elif self.FLAGS.local_rank in [2,6]:#back
                rotate_y = np.random.uniform( np.deg2rad(135), np.deg2rad(225))
            elif self.FLAGS.local_rank in [3,7]:
                rotate_y = np.random.uniform(np.deg2rad(-135), np.deg2rad(-45)) 
            if rotate_y > np.pi:
                rotate_y = rotate_y - np.pi*2
        elif self.FLAGS.gpu_number == 4: #All the results in the paper were generated using 8 3090 GPUs. We cannot guarantee that fewer than 8 GPUs can achieve the same effect.
            if self.FLAGS.local_rank in [0]:
                rotate_y = np.random.uniform(np.deg2rad(-45), np.deg2rad(45))
            elif self.FLAGS.local_rank in [1]:
                rotate_y = np.random.uniform(np.deg2rad(45), np.deg2rad(135))
            elif self.FLAGS.local_rank in [2]:#back
                rotate_y = np.random.uniform( np.deg2rad(135), np.deg2rad(225))
            elif self.FLAGS.local_rank in [3]:
                rotate_y = np.random.uniform(np.deg2rad(-135), np.deg2rad(-45)) 
            if rotate_y > np.pi:
                rotate_y = rotate_y - np.pi*2
        else:
            rotate_y = np.random.uniform(np.deg2rad(-180), np.deg2rad(180)) #All the results in the paper were generated using 8 3090 GPUs. We cannot guarantee that fewer than 8 GPUs can achieve the same effect.
    
        rotate_x = -np.random.uniform(self.elevation_range_min, self.elevation_range_max)
        angle_front = np.deg2rad(45)
        prompt_index = get_view_direction(thetas= rotate_x, phis = rotate_y, front= angle_front)
        cam_radius = 3.0
        x = np.random.uniform(-self.FLAGS.camera_random_jitter, self.FLAGS.camera_random_jitter)
        y = np.random.uniform(-self.FLAGS.camera_random_jitter, self.FLAGS.camera_random_jitter)
        mv     = util.translate(x, y, -cam_radius) @ (util.rotate_x(rotate_x) @ util.rotate_y(rotate_y))
        if ((itr+1)/self.FLAGS.batch) <=self.FLAGS.coarse_iter:
            rotate_y1 = np.random.uniform(0,np.pi*2)      
            rotate_x1 = np.random.uniform(-np.pi,np.pi)
            normal_rotate =  util.rotate_y_1(rotate_y1 ) @ util.rotate_x_1(rotate_x1)   
        else:
            normal_rotate =  util.rotate_y_1(0)@util.rotate_x_1(0)
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]
        ###
        # iter_res = self.FLAGS.train_res
        # proj_mtx = util.perspective(np.deg2rad(45), iter_res[1] / iter_res[0], 0.1, 1000.0)
        # mv     = util.translate(0, 0, -cam_radius) @ util.random_rotation_translation(0.25)
        # mvp    = proj_mtx @ mv
        # campos = torch.linalg.inv(mv)[:3, 3]
        # normal_rotate = torch.eye(3, dtype=torch.float32, device='cuda')
        ###
        # print('normal_rotate:', normal_rotate)
        return mv[None, ...], mvp[None, ...], campos[None, ...], self.FLAGS.display_res, self.FLAGS.spp, normal_rotate[None,...], prompt_index

    def __len__(self):
        if self.gif == True:
            return 100
        else:
            return 4 if self.validate else (self.FLAGS.iter + 1) * self.FLAGS.batch

    def __getitem__(self, itr):
        if self.gif:
            mv, mvp, campos, iter_res, iter_spp, normal_rotate, prompt_index = self._gif_scene(itr)
        elif self.validate:
            mv, mvp, campos, iter_res, iter_spp, normal_rotate, prompt_index = self._validate_scene(itr)
        else:
            mv, mvp, campos, iter_res, iter_spp, normal_rotate, prompt_index = self._train_scene(itr)
            
        img = render.render_mesh(self.glctx, self.ref_mesh, mvp.cuda(), campos.cuda(), self.envlight, iter_res, spp=iter_spp, 
                                num_layers=self.FLAGS.layers, msaa=True, background=None,
                                #后面参数为fantasia比nvidiffrec多出部分
                                if_normal = False, normal_rotate = normal_rotate.cuda(), mode = self.FLAGS.mode,
                                if_flip_the_normal = False)['shaded']
        #, if_use_bump = False
        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : iter_res,
            'spp' : iter_spp,
            'normal_rotate': normal_rotate,
            'prompt_index' : prompt_index,
            'img' : img #加入真实图片监督
        }
    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        return {
            'mv' : torch.cat(list([item['mv'] for item in batch]), dim=0),
            'mvp' : torch.cat(list([item['mvp'] for item in batch]), dim=0),
            'campos' : torch.cat(list([item['campos'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp' : iter_spp,
            'normal_rotate' : torch.cat(list([item['normal_rotate'] for item in batch]), dim=0),
            # 'prompt_index' : torch.cat(list([item['prompt_index'] for item in batch]), dim=0),
            'prompt_index' : np.array([item['prompt_index'] for item in batch], dtype=np.int32),
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0),#加入真实图片监督
        }

@torch.no_grad()
def get_view_direction(thetas, phis, front):
    #                   phis [B,];  -pi~pi        thetas: [B,] -pi/2~pi/2 
    # front = 0         [-front, front) 
    # side (left) = 1   [front, pi - front)
    # back = 2          [pi - front, pi) or [-pi, -pi+front)
    # side (right) = 3  [-pi+front, - front)
    
    if (phis >= -front) and (phis < front) :
        prompt_index = 0
    elif  (phis >= front ) and (phis < np.pi - front ):
        prompt_index = 1
    elif (phis >= np.pi - front) or  (phis < -np.pi + front):
        prompt_index = 2
    elif (phis >= -np.pi + front) and (phis < -front):
        prompt_index = 3
    
    return prompt_index

    
