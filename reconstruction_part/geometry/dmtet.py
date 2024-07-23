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

from render import mesh
from render import render
from render import util
import kaolin
from geometry.dmtet_network import Decoder
from render import regularizer
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd 
import cubvh, tqdm, trimesh

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None
    
class DMTet:
    def __init__(self):
        self.triangle_table = torch.tensor([   
                [-1, -1, -1, -1, -1, -1],     
                [ 1,  0,  2, -1, -1, -1],     
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],     
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long).cuda()   
        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device='cuda') #三角面片数量表，一共有16种类型的四面体，每一种四面体都对应要提取出的三角面片数量
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device='cuda') #基本的边，每一个数字是顶点索引，总共有12条边

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda")
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0 
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4) 
            occ_sum = torch.sum(occ_fx4, -1) 
            valid_tets = (occ_sum>0) & (occ_sum<4) 
            occ_sum = occ_sum[valid_tets] 
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2) 
            all_edges = self.sort_edges(all_edges) 
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)    
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1 
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1 
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda") 
            idx_map = mapping[idx_map] 
            interp_v = unique_edges[mask_edges] 
            
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3) 
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1) 
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True) 
        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator 
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1) 
        idx_map = idx_map.reshape(-1,6)
        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda")) 
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1) 
        num_triangles = self.num_triangles_table[tetindex] 

        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)
       
        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0] 
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets] 
        face_gidx = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0)

        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)

        return verts, faces, uvs, uv_idx

###############################################################################
#  Geometry interface
###############################################################################

class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, FLAGS):
        super(DMTetGeometry, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.marching_tets = DMTet()
 
        # self.decoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))
        tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))
        self.verts    =  torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale
        print("tet grid min/max", torch.min(self.verts).item(), torch.max(self.verts).item())
        self.decoder = Decoder(multires=0 , AABB= self.getAABB(), mesh_scale= scale)
        self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.generate_edges()
        self.initial_shape(FLAGS)

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2) 
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values
    
    def read_initial_shape(self, FLAGS):
        # base mesh
        mesh = trimesh.load(FLAGS.base_mesh, force='mesh')
        scale = 1 / np.array(mesh.bounds[1] - mesh.bounds[0]).max()
        center = np.array(mesh.bounds[1] + mesh.bounds[0]) / 2
        faces = mesh.faces
        vertices = (mesh.vertices - center) *  scale# N, 3

        # assert len(FLAGS.base_mesh_scale) == len(FLAGS.base_mesh_offset) == len(FLAGS.base_mesh_rotate_x)
        bs = len(FLAGS.base_mesh_scale)
        N = len(vertices)
        vertices = np.tile(vertices, (bs, 1, 1))
        
        # scale
        scale_array = np.array(FLAGS.base_mesh_scale) # bs, 3
        scale_array = scale_array[:, np.newaxis, :]
        scale_array = np.repeat(scale_array, N, axis=1)
        vertices = vertices * scale_array

        # rotate
        for i in range(bs):
            vertices[i] = vertices[i] @ util.rotate_x_2(np.deg2rad(FLAGS.base_mesh_rotate_x[i])) @ util.rotate_z_2(np.deg2rad(FLAGS.base_mesh_rotate_z[i]))

        # offset
        offset_array = np.array(FLAGS.base_mesh_offset) # bs, 3
        offset_array = offset_array[:, np.newaxis, :]
        offset_array = np.repeat(offset_array, N, axis=1)
        vertices = vertices + offset_array

        return vertices, faces

    def initial_shape(self, FLAGS):
        vertices, faces = self.read_initial_shape(FLAGS) # v [bs, N, 3], faces [F, 3]

        base_sdf_list = []
        for i in range(len(vertices)):
            BVH = cubvh.cuBVH(vertices[i], faces) # build with numpy.ndarray/torch.Tensor
            sdf, face_id, _ = BVH.signed_distance(self.verts, return_uvw=False, mode='raystab')
            sdf *= -1 
            base_sdf_list.append(sdf)
        base_sdf_list = torch.stack(base_sdf_list)
        merged_sdf, _ = torch.max(base_sdf_list, dim=0)
        merged_sdf = merged_sdf.detach().clone()

        # pretraining 
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)
        
        pretrain_iters = 15000    # 15000
        batch_size = 10240
        print(f"[INFO] start SDF pre-training ")
        for i in tqdm.tqdm(range(pretrain_iters)):
            rand_idx = torch.randint(0, self.verts.shape[0], (batch_size,))
            p = self.verts[rand_idx]
            ref_value = merged_sdf[rand_idx]
            output = self.decoder(p)
            loss = loss_fn(output[...,0], ref_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[INFO] SDF pre-train final loss: {loss.item()}")

    def getMesh(self, material):
        pred= self.decoder(self.verts)
        
        self.sdf , self.deform =  pred[:, 0], pred[:, 1:]  
        v_deformed = self.verts + 1 / (self.grid_res ) * torch.tanh(self.deform)
        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, self.sdf, self.indices)
       
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None, if_normal=False, mode = 'geometry_modeling', if_flip_the_normal = False):
        if mode == 'geometry_modeling':
            opt_mesh = self.getMesh(opt_material) 
        else:
            with torch.no_grad():
                opt_mesh = self.getMesh(opt_material) 
        return render.render_mesh(glctx, 
                                  opt_mesh, 
                                  target['mvp'], 
                                  target['campos'], 
                                  lgt, 
                                  target['resolution'], 
                                  spp=target['spp'], 
                                  msaa= True,
                                  background= target['background'],
                                  bsdf= bsdf,
                                  if_normal= if_normal,
                                  normal_rotate= target['normal_rotate'],
                                  mode = mode,
                                  if_flip_the_normal = if_flip_the_normal
                                  )

        
    def tick(self, glctx, target, lgt, opt_material, iteration, if_normal, guidance, mode, if_flip_the_normal):
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers= self.render(glctx, target, lgt, opt_material, if_normal= if_normal, mode = mode, if_flip_the_normal = if_flip_the_normal)
        if self.FLAGS.add_directional_text:
            text_embeddings = torch.cat([guidance.uncond_z[target['prompt_index']], guidance.text_z[target['prompt_index']]]) # [B*2, 77, 1024]
        else:
            text_embeddings = torch.cat([guidance.uncond_z, guidance.text_z])  # [B * 2, 77, 1024]

        if mode == 'geometry_modeling':
            return self.geo_tick(iteration, guidance, buffers, text_embeddings)
        else: 
            return self.appear_tick(iteration, guidance, buffers, text_embeddings)

        
    def geo_tick(self, iteration, guidance, buffers, text_embeddings):

        if iteration <=self.FLAGS.coarse_iter:
            t = torch.randint( guidance.min_step_early, guidance.max_step_early + 1, [self.FLAGS.batch], dtype=torch.long, device='cuda') # [B]
            # t = torch.randint( guidance.min_step_early, guidance.max_step_early + 1, [1], dtype=torch.long, device='cuda') # [B]
            latents =  buffers['shaded'][..., 0:4].permute(0, 3, 1, 2).contiguous() # [B, 4, 64, 64]
            latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
         
        else:
            t = torch.randint(guidance.min_step_late, guidance.max_step_late + 1, [self.FLAGS.batch], dtype=torch.long, device='cuda')
            # t = torch.randint(guidance.min_step_late, guidance.max_step_late + 1, [1], dtype=torch.long, device='cuda')
            srgb =  buffers['shaded'][...,0:3] #* buffers['shaded'][..., 3:4] # normal * mask
            # 
            pred_rgb_512 = srgb.permute(0, 3, 1, 2).contiguous() # [B, 3, 512, 512]
            latents = guidance.encode_imgs(pred_rgb_512)
   
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = guidance.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = guidance.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred =noise_pred_uncond + guidance.guidance_weight * (noise_pred_text - noise_pred_uncond) # [B, 4, 64, 64]
        
        # w = (1 - guidance.alphas[t]) # [B]
        # w = self.guidance.alphas[t]
        # w = 1 / (1 - guidance.alphas[t])
        # w = 1 / torch.sqrt(1 - guidance.alphas[t])
        w = guidance.alphas[t] ** 0.5 * (1 - guidance.alphas[t])
        w = w[:, None, None, None] # [B, 1, 1, 1]
        grad =  w * (noise_pred - noise ) #*w1 
        grad = torch.nan_to_num(grad)
        
        sds_loss = SpecifyGradient.apply(latents, grad)     
        img_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        return sds_loss, img_loss, reg_loss


    def appear_tick(self, iteration, guidance, buffers, text_embeddings):
        if iteration <= self.FLAGS.coarse_iter:
            srgb =  buffers['shaded'][...,0:3]
            srgb = util.rgb_to_srgb(srgb)
            # t = torch.randint( guidance.min_step_early, guidance.max_step_early, [1], dtype=torch.long, device='cuda')
            t = torch.randint( guidance.min_step_early, guidance.max_step_early+1, [self.FLAGS.batch], dtype=torch.long, device='cuda') # [B]
        else:
            srgb =   buffers['shaded'][...,0:3]
            srgb = util.rgb_to_srgb(srgb)
            # t = torch.randint(guidance.min_step_late, guidance.max_step_late, [1], dtype=torch.long, device='cuda')
            t = torch.randint( guidance.min_step_late, guidance.max_step_late+1, [self.FLAGS.batch], dtype=torch.long, device='cuda') # [B]

        pred_rgb_512 = srgb.permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
        latents = guidance.encode_imgs(pred_rgb_512)
       
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = guidance.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = guidance.unet(latent_model_input, tt, encoder_hidden_states= text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance.guidance_weight * (noise_pred_text - noise_pred_uncond)
    
        if guidance.sds_weight_strategy == 0:
            w = guidance.alphas[t] ** 0.5 * (1 - guidance.alphas[t])
        elif guidance.sds_weight_strategy == 1:
            # w = 1 / torch.sqrt(1 - guidance.alphas[t])
            # w = (1 - guidance.alphas[t]) 
            w = 1 / (1 - guidance.alphas[t])
        elif guidance.sds_weight_strategy == 2:
            if iteration <= self.FLAGS.coarse_iter:
                w = guidance.alphas[t] ** 0.5 * (1 - guidance.alphas[t])
            else:
                w = 1 / (1 - guidance.alphas[t])
        w = w[:, None, None, None] # [B, 1, 1, 1]
        grad = w* (noise_pred -noise) 
        grad = torch.nan_to_num(grad)
        sds_loss = SpecifyGradient.apply(latents, grad) 
        img_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        # reg_loss = torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) *10
     
        return sds_loss, img_loss, reg_loss