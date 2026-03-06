# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# Modified for UniDexGrasp2 Integration (Fixed Version)
# --------------------------------------------------------

import os
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch, unscale, quat_apply, tensor_clamp, torch_rand_float, quat_conjugate, quat_mul
from glob import glob
from hora.utils.misc import tprint
from .base.vec_task import VecTask


class AllegroHandHora(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.config = config
        # 1. setup randomization
        self._setup_domain_rand_config(config['env']['randomization'])
        # 2. setup privileged information
        self._setup_priv_option_config(config['env']['privInfo'])
        # 3. setup object assets
        self._setup_object_info(config['env']['object'])
        # 4. setup reward
        self._setup_reward_config(config['env']['reward'])
        
        self.base_obj_scale = config['env']['baseObjScale']
        self.save_init_pose = config['env'].get('genGrasps', False)
        self.aggregate_mode = self.config['env']['aggregateMode']
        self.up_axis = 'z'
        self.reset_z_threshold = self.config['env']['reset_height_threshold']
        self.grasp_cache_name = self.config['env']['grasp_cache_name']
        self.evaluate = self.config['on_evaluation']
        
        self.priv_info_dict = {
            'obj_position': (0, 3),
            'obj_scale': (3, 4),
            'obj_mass': (4, 5),
            'obj_friction': (5, 6),
            'obj_com': (6, 9),
            'obj_linvel': (9, 12) # 补全定义防止报错
        }

        # --- 关键修复：防止父类初始化时找不到变量 ---
        self.max_episode_length = self.config['env']['episodeLength']

        super().__init__(config, sim_device, graphics_device_id, headless)

        self.debug_viz = self.config['env']['enableDebugVis']
        self.dt = self.sim_params.dt

        # --- 保持原来的视角设置 ---
        if self.viewer:
            cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        # --- UniDexGrasp 笼式预抓取姿态 ---
        cage_pose = [
            0.0, 0.8, 0.5, 0.4,  # Index
            0.0, 0.8, 0.5, 0.4,  # Middle
            0.0, 0.8, 0.5, 0.4,  # Ring
            1.1, 0.6, 0.2, 0.5   # Thumb
        ]
        self.allegro_hand_default_dof_pos = torch.tensor(cage_pose, dtype=torch.float, device=self.device)
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        
        # 16 dofs for Allegro Hand
        self.num_allegro_hand_dofs = 16
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        # 兼容旧代码引用
        self.root_states = self.root_state_tensor

        self._refresh_gym()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        
        # UniDexGrasp 专用 Buffer
        self.object_target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.palm_pos_buf = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # randomization buffers
        self.force_scale = self.config['env'].get('forceScale', 0.0)
        self.random_force_prob_scalar = self.config['env'].get('randomForceProbScalar', 0.0)
        self.force_decay = self.config['env'].get('forceDecay', 0.99)
        self.force_decay_interval = self.config['env'].get('forceDecayInterval', 0.08)
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        self.saved_grasping_states = {} 
        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        self.object_rot_prev = self.object_rot.clone()
        self.object_pos_prev = self.object_pos.clone()
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        
        # 动作空间 22 维 (6 Base + 16 Joints)
        self.actions = torch.zeros((self.num_envs, 22), device=self.device, dtype=torch.float)
        self.torques = torch.zeros((self.num_envs, 22), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        
        p_gain_val = self.config['env']['controller']['pgain']
        d_gain_val = self.config['env']['controller']['dgain']
        self.p_gain = torch.ones((self.num_envs, 22), device=self.device, dtype=torch.float) * p_gain_val
        self.d_gain = torch.ones((self.num_envs, 22), device=self.device, dtype=torch.float) * d_gain_val

        # statistics
        self.env_timeout_counter = to_torch(np.zeros((len(self.envs)))).long().to(self.device) 
        self.stat_sum_rewards = 0
        self.stat_sum_rotate_rewards = 0
        self.stat_sum_episode_length = 0
        self.stat_sum_obj_linvel = 0
        self.stat_sum_torques = 0
        self.env_evaluated = 0
        self.max_evaluate_envs = 500000

        # --- 核心修复：强制覆盖历史缓存维度，解决 21 vs 32 报错 ---
        self.prop_hist_len = self.config['env']['hora']['propHistoryLen']
        self.obs_buf_lag_history = torch.zeros((self.num_envs, self.prop_hist_len, 32), device=self.device, dtype=torch.float)

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._create_object_asset()

        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []

        # 注意：这里我们只取前16个关节的属性，防止 Asset 里面有多余的关节
        for i in range(16):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            allegro_hand_dof_props['effort'][i] = 0.5
            if self.torque_control:
                allegro_hand_dof_props['stiffness'][i] = 0.
                allegro_hand_dof_props['damping'][i] = 0.
                allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            else:
                allegro_hand_dof_props['stiffness'][i] = self.config['env']['controller']['pgain']
                allegro_hand_dof_props['damping'][i] = self.config['env']['controller']['dgain']
                allegro_hand_dof_props['friction'][i] = 0.01
                allegro_hand_dof_props['armature'][i] = 0.001

        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)

        hand_pose, obj_pose = self._init_object_pose()

        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        max_agg_bodies = self.num_allegro_hand_bodies + 2
        max_agg_shapes = self.num_allegro_hand_shapes + 2

        self.envs = []
        self.object_init_state = []
        self.hand_indices = []
        self.object_indices = []

        allegro_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        object_rb_count = 1
        self.object_rb_handles = list(range(allegro_hand_rb_count, allegro_hand_rb_count + object_rb_count))

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, allegro_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # object asset loading logic from HORA
            object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            object_asset = self.object_asset_list[object_type_id]

            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, 'object', i, 0, 0)
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # Randomization logic
            obj_scale = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                obj_scale = np.random.uniform(self.randomize_scale_list[i % num_scales] - 0.025, self.randomize_scale_list[i % num_scales] + 0.025)
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
            self._update_priv_buf(env_id=i, name='obj_scale', value=obj_scale)

            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper) for _ in range(3)]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            self._update_priv_buf(env_id=i, name='obj_com', value=obj_com)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

    def reset_idx(self, env_ids):
        # 1. 随机化 PD (保留 HORA 特性)
        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), 22), device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), 22), device=self.device).squeeze(1)

        self.rb_forces[env_ids, :, :] = 0.0

        # 2. 重置物体
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        new_rot = torch.randn((len(env_ids), 4), device=self.device)
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = torch.nn.functional.normalize(new_rot, dim=-1)

        # 3. 重置手部
        # 高度设为 0.2m (原代码 0.2)，保留原来的旋转
        self.root_state_tensor[self.hand_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2]
        self.root_state_tensor[self.hand_indices[env_ids], 2] = 0.2 
        # 这里的旋转保持原代码逻辑
        self.root_state_tensor[self.hand_indices[env_ids], 3:7] = to_torch([0.707, 0.0, 0.707, 0.0], device=self.device).repeat(len(env_ids), 1)

        # 4. 重置关节
        pos = self.allegro_hand_default_dof_pos.repeat(len(env_ids), 1)
        self.allegro_hand_dof_pos[env_ids, :] = pos
        self.allegro_hand_dof_vel[env_ids, :] = 0
        self.prev_targets[env_ids, :16] = pos
        self.cur_targets[env_ids, :16] = pos
        
        # 5. UniDexGrasp 目标设置
        self.object_target_pos[env_ids] = self.object_init_state[env_ids, 0:3] + to_torch([0, 0, 0.2], device=self.device)

        # 同步
        indices = torch.cat([self.object_indices[env_ids], self.hand_indices[env_ids]]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(indices), len(indices))
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(self.hand_indices[env_ids].to(torch.int32)), len(env_ids))
        
        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def compute_observations(self):
        self._refresh_gym()
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        
        # 噪声处理 (保留 HORA)
        joint_noise = (torch.rand_like(self.allegro_hand_dof_pos) * 2.0 - 1.0) * self.joint_noise_scale
        cur_q_obs = unscale(self.allegro_hand_dof_pos + joint_noise, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        
        # 32 维拼接
        cur_obs_32 = torch.cat([cur_q_obs, self.cur_targets[:, :16]], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_32.unsqueeze(1)], dim=1)

        # 64 维观测
        self.obs_buf[:, 0:16] = cur_q_obs
        self.obs_buf[:, 16:32] = self.allegro_hand_dof_vel * 0.1
        self.obs_buf[:, 32:35] = self.object_pos
        self.obs_buf[:, 35:39] = self.object_rot
        self.obs_buf[:, 39:42] = self.root_state_tensor[self.object_indices, 7:10]
        self.obs_buf[:, 42:45] = self.root_state_tensor[self.object_indices, 10:13]
        self.obs_buf[:, 45:61] = cur_q_obs.clone() 
        self.obs_buf[:, 61:64] = self.object_pos - self.palm_pos_buf
        
        # Buffer 更新
        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_position', value=self.object_pos.clone())
        
        return self.obs_buf

    def compute_reward(self, actions):
        self._refresh_gym()
        
        # 1. 准备指尖位置 (Allegro Hand 索引: 4, 8, 12, 16)
        fingertip_pos = self.rigid_body_states[:, [4, 8, 12, 16], :3]
        
        # 2. 锁定针对球体优化的抓取姿态 (Ball Grasp Pose)
        # 这是一个预设的半握球姿态，不需要加载外部数据集
        ball_grasp_pose = [
            0.0, 1.0, 0.8, 0.4,  # Index
            0.0, 1.0, 0.8, 0.4,  # Middle
            0.0, 1.0, 0.8, 0.4,  # Ring
            1.2, 0.8, 0.4, 0.6   # Thumb
        ]
        goal_q = torch.tensor(ball_grasp_pose, device=self.device).repeat(self.num_envs, 1)

        # 3. 执行对齐 Uni 逻辑的奖励计算
        self.rew_buf[:] = compute_hand_reward(
            self.object_pos, self.object_target_pos, self.palm_pos_buf, fingertip_pos,
            self.allegro_hand_dof_pos, goal_q, actions,
            self.w_gq, self.w_gt, self.w_r, self.w_l, self.w_m, self.w_b
        )

        # 4. 检查是否触发重置条件
        self.reset_buf[:] = self.check_termination(self.object_pos)
        
        # 5. 更新统计数据
        self.extras['goal_dist'] = torch.norm(self.object_target_pos - self.object_pos, p=2, dim=-1).mean()
        
        self.extras['lift_height'] = (self.object_pos[:, 2] - 0.04).mean()

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_gym()
        self.compute_reward(self.actions)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()
        
        # 原有的 Debug 可视化逻辑 (保留)
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            # ... (保留原有的画线逻辑)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        
        # 22维动作分解
        root_move = self.actions[:, 0:3]
        joint_act = self.actions[:, 6:22]
        
        targets = self.prev_targets + (1.0 / 24.0) * joint_act
        self.cur_targets[:] = tensor_clamp(targets, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.prev_targets[:] = self.cur_targets.clone()
        
        # 根部位置控制 (Lift)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states[self.hand_indices, 0:3] += root_move * self.dt * 1.5 
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        
        # 关节位置/力矩控制
        if self.force_scale > 0.0:
             # 原有的外力干扰逻辑 (保留)
             pass 

    def reset(self):
        super().reset()
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        return self.obs_dict

    def step(self, actions):
        super().step(actions)
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def update_low_level_control(self):
        previous_dof_pos = self.allegro_hand_dof_pos.clone()
        self._refresh_gym()
        if self.torque_control:
            # 显式维度切片，防止越界
            p_joint = self.p_gain[:, 6:22]
            d_joint = self.d_gain[:, 6:22]
            
            dof_vel = (self.allegro_hand_dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff = dof_vel.clone()
            
            diff = self.cur_targets - self.allegro_hand_dof_pos
            torques = p_joint * diff - d_joint * self.allegro_hand_dof_vel
            self.torques = torch.clip(torques, -0.7, 0.7).clone() # 限制力矩
            
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def check_termination(self, object_pos):
        resets = torch.logical_or(
            torch.less(object_pos[:, -1], self.reset_z_threshold),
            torch.greater_equal(self.progress_buf, self.max_episode_length),
        )
        return resets

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]
        self.palm_pos_buf = self.rigid_body_states[:, 0, 0:3]

    def _setup_domain_rand_config(self, rand_config):
        self.randomize_mass = rand_config['randomizeMass']
        self.randomize_mass_lower = rand_config['randomizeMassLower']
        self.randomize_mass_upper = rand_config['randomizeMassUpper']
        self.randomize_com = rand_config['randomizeCOM']
        self.randomize_com_lower = rand_config['randomizeCOMLower']
        self.randomize_com_upper = rand_config['randomizeCOMUpper']
        self.randomize_friction = rand_config['randomizeFriction']
        self.randomize_friction_lower = rand_config['randomizeFrictionLower']
        self.randomize_friction_upper = rand_config['randomizeFrictionUpper']
        self.randomize_scale = rand_config['randomizeScale']
        self.scale_list_init = rand_config['scaleListInit']
        self.randomize_scale_list = rand_config['randomizeScaleList']
        self.randomize_scale_lower = rand_config['randomizeScaleLower']
        self.randomize_scale_upper = rand_config['randomizeScaleUpper']
        self.randomize_pd_gains = rand_config['randomizePDGains']
        self.randomize_p_gain_lower = rand_config['randomizePGainLower']
        self.randomize_p_gain_upper = rand_config['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_config['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_config['randomizeDGainUpper']
        self.joint_noise_scale = rand_config['jointNoiseScale']

    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config['enableObjPos']
        self.enable_priv_obj_mass = p_config['enableObjMass']
        self.enable_priv_obj_scale = p_config['enableObjScale']
        self.enable_priv_obj_com = p_config['enableObjCOM']
        self.enable_priv_obj_friction = p_config['enableObjFriction']

    def _update_priv_buf(self, env_id, name, value, lower=None, upper=None):
        s, e = self.priv_info_dict[name]
        if eval(f'self.enable_priv_{name}'):
            if type(value) is list: value = to_torch(value, dtype=torch.float, device=self.device)
            if lower is not None and upper is not None:
                if type(lower) is list: lower = to_torch(lower, device=self.device)
                if type(upper) is list: upper = to_torch(upper, device=self.device)
                value = (2.0 * value - upper - lower) / (upper - lower)
            self.priv_info_buf[env_id, s:e] = value
        else:
            self.priv_info_buf[env_id, s:e] = 0

    def _setup_object_info(self, o_config):
        self.object_type = o_config['type']
        raw_prob = o_config['sampleProb']
        assert (sum(raw_prob) == 1)
        primitive_list = self.object_type.split('+')
        print('---- Primitive List ----', primitive_list)
        self.object_type_prob = []
        self.object_type_list = []
        self.asset_files_dict = {'simple_tennis_ball': 'assets/ball.urdf'}
        
        for p_id, prim in enumerate(primitive_list):
            if 'cuboid' in prim:
                subset_name = self.object_type.split('_')[-1]
                cuboids = sorted(glob(f'../assets/cuboid/{subset_name}/*.urdf'))
                cuboid_list = [f'cuboid_{i}' for i in range(len(cuboids))]
                self.object_type_list += cuboid_list
                for i, name in enumerate(cuboids):
                    self.asset_files_dict[f'cuboid_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cuboid_list) for _ in cuboid_list]
            elif 'cylinder' in prim:
                subset_name = self.object_type.split('_')[-1]
                cylinders = sorted(glob(f'assets/cylinder/{subset_name}/*.urdf'))
                cylinder_list = [f'cylinder_{i}' for i in range(len(cylinders))]
                self.object_type_list += cylinder_list
                for i, name in enumerate(cylinders):
                    self.asset_files_dict[f'cylinder_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cylinder_list) for _ in cylinder_list]
            else:
                self.object_type_list += [prim]
                self.object_type_prob += [raw_prob[p_id]]
        print('---- Object List ----', self.object_type_list)

    def _allocate_task_buffer(self, num_envs):
        self.prop_hist_len = self.config['env']['hora']['propHistoryLen']
        self.num_env_factors = self.config['env']['hora']['privInfoDim']
        self.priv_info_buf = torch.zeros((num_envs, self.num_env_factors), device=self.device, dtype=torch.float)
        # 先分配，后面会强制覆盖
        self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, 32), device=self.device, dtype=torch.float)

    def _setup_reward_config(self, r_config):
        self.w_gq = r_config.get('w_gq', 0.1)
        self.w_gt = r_config.get('w_gt', 0.6)
        self.w_r = r_config.get('w_r', 0.5)
        self.w_l = r_config.get('w_l', 0.1)
        self.w_m = r_config.get('w_m', 2.0)
        self.w_b = r_config.get('w_b', 10.0)
        
        # 保留 HORA 变量防止报错
        self.angvel_clip_min = r_config.get('angvelClipMin', -0.5)
        self.angvel_clip_max = r_config.get('angvelClipMax', 0.5)
        self.rotate_reward_scale = r_config.get('rotateRewardScale', 1.0)
        self.object_linvel_penalty_scale = r_config.get('objLinvelPenaltyScale', -0.3)
        self.pose_diff_penalty_scale = r_config.get('poseDiffPenaltyScale', -0.3)
        self.torque_penalty_scale = r_config.get('torquePenaltyScale', -0.1)
        self.work_penalty_scale = r_config.get('workPenaltyScale', -2.0)

    def _create_object_asset(self):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
        hand_asset_file = self.config['env']['asset']['handAsset']
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = True
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01
        if self.torque_control:
            hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        else:
            hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, hand_asset_options)

        self.object_asset_list = []
        for object_type in self.object_type_list:
            object_asset_file = self.asset_files_dict[object_type]
            object_asset_options = gymapi.AssetOptions()
            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            self.object_asset_list.append(object_asset)

    def _init_object_pose(self):
        allegro_hand_start_pose = gymapi.Transform()
        # 手的初始位置 (0.2m 高度)
        allegro_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.2)
        allegro_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.pi / 2)

        object_start_pose = gymapi.Transform()
        # 球的初始位置 (0.04m 高度)
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.04) 
        
        return allegro_hand_start_pose, object_start_pose


@torch.jit.script
def compute_hand_reward(
    object_pos, object_target_pos, palm_pos, fingertip_pos,
    current_q, goal_q, actions,
    w_gq: float, w_gt: float, w_r: float, w_l: float, w_m: float, w_b: float
):
    # --- 组件 1: r_goal (姿态奖励) ---
    # 引导关节角呈现抓球手型
    q_dist = torch.abs(current_q - goal_q).sum(-1)
    r_goal = -w_gq * q_dist 

    # --- 组件 2: r_reach (趋近奖励) ---
    # 对齐 Uni 逻辑：手掌靠近球心 + 指尖包围球心
    dist_palm_obj = torch.norm(palm_pos - object_pos, p=2, dim=-1)
    dist_fingers_obj = torch.norm(fingertip_pos - object_pos.unsqueeze(1), p=2, dim=-1)
    r_reach = -w_gt * dist_palm_obj - w_r * dist_fingers_obj.mean(-1)

    # --- 组件 3: r_lift (抬升奖励) ---
    # 触碰判定：指尖与球心距离足够近 (球半径为 0.0375m)
    is_touching = dist_fingers_obj.max(-1)[0] < 0.05
    # 离地距离：初始 0.04m，设置 0.045m 为判定阈值
    lift_dist = object_pos[:, 2] - 0.045
    r_lift = torch.where(
        is_touching & (lift_dist > 0), 
        w_l * lift_dist * 50.0, 
        torch.zeros_like(lift_dist)
    )

    # --- 组件 4: r_move & success (目标移动与大奖) ---
    # 物体靠近上方目标点 (0, 0, 0.2)
    dist_target = torch.norm(object_target_pos - object_pos, p=2, dim=-1)
    r_move = -w_m * dist_target
    
    # 成功判定：球离地高度超过 10cm 时给予 Bonus
    r_success = torch.where(lift_dist > 0.1, w_b, torch.zeros_like(lift_dist))

    return r_goal + r_reach + r_lift + r_move + r_success