
import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from .base.vec_task import VecTask


class FrankaColorStack(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1/60.

        self.prop_width = 0.08
        self.prop_height = 0.08
        self.prop_length = 0.08
        self.prop_spacing = 0.09

        num_obs = 36
        num_acts = 9

        self.cfg["env"]["numObservations"] = 36
        self.cfg["env"]["numActions"] = 9

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.franka_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]
        self.cubeA_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:self.num_franka_dofs+6]
        self.cubeA_dof_pos = self.cubeA_dof_state[..., 0]
        self.cubeA_dof_vel = self.cubeA_dof_state[..., 1]
        self.cubeB_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs+6:self.num_franka_dofs+12]
        self.cubeB_dof_pos = self.cubeB_dof_state[..., 0]
        self.cubeB_dof_vel = self.cubeB_dof_state[..., 1]
        self.cubeC_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs+12:self.num_franka_dofs+18]
        self.cubeC_dof_pos = self.cubeC_dof_state[..., 0]
        self.cubeC_dof_vel = self.cubeC_dof_state[..., 1]
        self.cubeD_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs+18:self.num_franka_dofs+24]
        self.cubeD_dof_pos = self.cubeD_dof_state[..., 0]
        self.cubeD_dof_vel = self.cubeD_dof_state[..., 1]
        self.cubeE_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs+24:self.num_franka_dofs+30]
        self.cubeE_dof_pos = self.cubeE_dof_state[..., 0]
        self.cubeE_dof_vel = self.cubeE_dof_state[..., 1]
        self.cubeF_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs+24:self.num_franka_dofs+30]
        self.cubeF_dof_pos = self.cubeF_dof_state[..., 0]
        self.cubeF_dof_vel = self.cubeF_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        if self.num_props > 0:
            self.prop_states = self.root_state_tensor[:, 2:]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        cube_asset_file = "urdf/objects/cube_multicolor.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
            cube_asset_file = self.cfg["env"]["asset"].get("assetFileNameCube", cube_asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        cubeA_asset = self.gym.load_asset(self.sim, asset_root, cube_asset_file, asset_options)
        cubeB_asset = self.gym.load_asset(self.sim, asset_root, cube_asset_file, asset_options)
        cubeC_asset = self.gym.load_asset(self.sim, asset_root, cube_asset_file, asset_options)
        cubeD_asset = self.gym.load_asset(self.sim, asset_root, cube_asset_file, asset_options)
        cubeE_asset = self.gym.load_asset(self.sim, asset_root, cube_asset_file, asset_options)
        cubeF_asset = self.gym.load_asset(self.sim, asset_root, cube_asset_file, asset_options)

        franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.num_cube_bodies = self.gym.get_asset_rigid_body_count(cubeA_asset)+self.gym.get_asset_rigid_body_count(cubeB_asset)+self.gym.get_asset_rigid_body_count(cubeC_asset)+self.gym.get_asset_rigid_body_count(cubeD_asset)+self.gym.get_asset_rigid_body_count(cubeE_asset)
        self.num_cube_dofs = self.gym.get_asset_dof_count(cubeA_asset)+self.gym.get_asset_dof_count(cubeA_asset)+self.gym.get_asset_dof_count(cubeB_asset)+self.gym.get_asset_dof_count(cubeC_asset)+self.gym.get_asset_dof_count(cubeD_asset)+self.gym.get_asset_dof_count(cubeE_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)
        print("num cube bodies: ", self.num_cube_bodies)
        print("num cube dofs: ", self.num_cube_dofs)

        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        cubeA_dof_props = self.gym.get_asset_dof_properties(cubeA_asset)
        cubeB_dof_props = self.gym.get_asset_dof_properties(cubeB_asset)
        cubeC_dof_props = self.gym.get_asset_dof_properties(cubeC_asset)
        cubeD_dof_props = self.gym.get_asset_dof_properties(cubeD_asset)
        cubeE_dof_props = self.gym.get_asset_dof_properties(cubeE_asset)
        cubeF_dof_props = self.gym.get_asset_dof_properties(cubeF_asset)
        for i in range(self.num_cube_dofs):
            cube_dof_props['damping'][i] = 10.0

        box_opts = gymapi.AssetOptions()
        box_opts.density = 400
        prop_asset = self.gym.create_box(self.sim, self.prop_width, self.prop_height, self.prop_width, box_opts)

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, self.up_axis_idx))

        cubeB_start_pose = gymapi.Transform()
        cubeB_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, self.up_axis_idx))
        cubeC_start_pose = gymapi.Transform()
        cubeC_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, self.up_axis_idx))
        cubeD_start_pose = gymapi.Transform()
        cubeD_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, self.up_axis_idx))
        cubeE_start_pose = gymapi.Transform()
        cubeE_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, self.up_axis_idx))

        cubeF_start_pose = gymapi.Transform()
        cubeF_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, self.up_axis_idx))

        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_cube_bodies = self.gym.get_asset_rigid_body_count(cubeA_asset)+self.gym.get_asset_rigid_body_count(cubeB_asset)+self.gym.get_asset_rigid_body_count(cubeC_asset)+self.gym.get_asset_rigid_body_count(cubeD_asset)+self.gym.get_asset_rigid_body_count(cubeE_asset)+self.gym.get_asset_rigid_body_count(cubeF_asset)
        num_cube_shapes = self.gym.get_asset_rigid_shape_count(cubeA_asset)+self.gym.get_asset_rigid_shape_count(cubeB_asset)+self.gym.get_asset_rigid_shape_count(cubeC_asset)+self.gym.get_asset_rigid_shape_count(cubeD_asset)+self.gym.get_asset_rigid_shape_count(cubeE_asset)+self.gym.get_asset_rigid_shape_count(cubeF_asset)
        num_prop_bodies = self.gym.get_asset_rigid_body_count(prop_asset)
        num_prop_shapes = self.gym.get_asset_rigid_shape_count(prop_asset)
        max_agg_bodies = num_franka_bodies + num_cube_bodies + self.num_props * num_prop_bodies
        max_agg_shapes = num_franka_shapes + num_cube_shapes + self.num_props * num_prop_shapes

        self.frankas = []
        self.cubes = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            cubeA_pose = cubeA_start_pose
            cubeA_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            cubeA_pose.p.y += self.start_position_noise * dy
            cubeA_pose.p.z += self.start_position_noise * dz
            cubeA_actor = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_pose, "cubeA", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, cubeA_actor, cubeA_dof_props)

            cubeB_pose = cubeB_start_pose
            cubeB_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            cubeB_pose.p.y += self.start_position_noise * dy
            cubeB_pose.p.z += self.start_position_noise * dz
            cubeB_actor = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_pose, "cubeB", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, cubeB_actor, cubeB_dof_props)

            cubeC_pose = cubeC_start_pose
            cubeC_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            cubeC_pose.p.y += self.start_position_noise * dy
            cubeC_pose.p.z += self.start_position_noise * dz
            cubeC_actor = self.gym.create_actor(env_ptr, cubeC_asset, cubeC_pose, "cubeC", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, cubeC_actor, cubeC_dof_props)

            cubeD_pose = cubeD_start_pose
            cubeD_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            cubeD_pose.p.y += self.start_position_noise * dy
            cubeD_pose.p.z += self.start_position_noise * dz
            cubeD_actor = self.gym.create_actor(env_ptr, cubeD_asset, cubeD_pose, "cubeD", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, cubeD_actor, cubeD_dof_props)

            cubeE_pose = cubeE_start_pose
            cubeE_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            cubeE_pose.p.y += self.start_position_noise * dy
            cubeE_pose.p.z += self.start_position_noise * dz
            cubeE_actor = self.gym.create_actor(env_ptr, cubeE_asset, cubeE_pose, "cubeE", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, cubeE_actor, cubeE_dof_props)

            cubeF_pose = cubeF_start_pose
            cubeF_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            cubeF_pose.p.y += self.start_position_noise * dy
            cubeF_pose.p.z += self.start_position_noise * dz
            cubeF_actor = self.gym.create_actor(env_ptr, cubeF_asset, cubeF_pose, "cubeF", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, cubeF_actor, cubeF_dof_props)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.num_props > 0:
                self.prop_start.append(self.gym.get_sim_actor_count(self.sim))
                cubeA_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cubeA_actor, "cubeA_handle")
                cubeA_pose = self.gym.get_rigid_transform(env_ptr, cubeA_handle)

                cubeB_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cubeB_actor, "cubeB_handle")
                cubeB_pose = self.gym.get_rigid_transform(env_ptr, cubeB_handle)

                cubeC_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cubeC_actor, "cubeC_handle")
                cubeC_pose = self.gym.get_rigid_transform(env_ptr, cubeC_handle)

                cubeD_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cubeD_actor, "cubeD_handle")
                cubeD_pose = self.gym.get_rigid_transform(env_ptr, cubeD_handle)

                cubeE_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cubeE_actor, "cubeE_handle")
                cubeE_pose = self.gym.get_rigid_transform(env_ptr, cubeE_handle)

                cubeF_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cubeF_actor, "cubeF_handle")
                cubeF_pose = self.gym.get_rigid_transform(env_ptr, cubeF_handle)

                props_per_row = int(np.ceil(np.sqrt(self.num_props)))
                xmin = -0.5 * self.prop_spacing * (props_per_row - 1)
                yzmin = -0.5 * self.prop_spacing * (props_per_row - 1)

                prop_count = 0
                for j in range(props_per_row):
                    prop_up = yzmin + j * self.prop_spacing
                    for k in range(props_per_row):
                        if prop_count >= self.num_props:
                            break
                        propx = xmin + k * self.prop_spacing
                        prop_state_pose = gymapi.Transform()
                        prop_state_pose.p.x = cubeA_pose.p.x + propx
                        propz, propy = 0, prop_up
                        prop_state_pose.p.y = cubeA_pose.p.y + propy
                        prop_state_pose.p.z = cubeA_pose.p.z + propz
                        prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
                        prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1

                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])

                        propx = xmin + k * self.prop_spacing
                        prop_state_pose = gymapi.Transform()
                        prop_state_pose.p.x = cubeB_pose.p.x + propx
                        propz, propy = 0, prop_up
                        prop_state_pose.p.y = cubeB_pose.p.y + propy
                        prop_state_pose.p.z = cubeB_pose.p.z + propz
                        prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
                        prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1

                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])
                        
                        propx = xmin + k * self.prop_spacing
                        prop_state_pose = gymapi.Transform()
                        prop_state_pose.p.x = cubeC_pose.p.x + propx
                        propz, propy = 0, prop_up
                        prop_state_pose.p.y = cubeC_pose.p.y + propy
                        prop_state_pose.p.z = cubeC_pose.p.z + propz
                        prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
                        prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1

                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])
                        
                        propx = xmin + k * self.prop_spacing
                        prop_state_pose = gymapi.Transform()
                        prop_state_pose.p.x = cubeD_pose.p.x + propx
                        propz, propy = 0, prop_up
                        prop_state_pose.p.y = cubeD_pose.p.y + propy
                        prop_state_pose.p.z = cubeD_pose.p.z + propz
                        prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
                        prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1

                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])

                        propx = xmin + k * self.prop_spacing
                        prop_state_pose = gymapi.Transform()
                        prop_state_pose.p.x = cubeE_pose.p.x + propx
                        propz, propy = 0, prop_up
                        prop_state_pose.p.y = cubeE_pose.p.y + propy
                        prop_state_pose.p.z = cubeE_pose.p.z + propz
                        prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
                        prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1

                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])

                        propx = xmin + k * self.prop_spacing
                        prop_state_pose = gymapi.Transform()
                        prop_state_pose.p.x = cubeF_pose.p.x + propx
                        propz, propy = 0, prop_up
                        prop_state_pose.p.y = cubeF_pose.p.y + propy
                        prop_state_pose.p.z = cubeF_pose.p.z + propz
                        prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
                        prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1

                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.cubes.append(cubeA_actor)
            self.cubes.append(cubeB_actor)
            self.cubes.append(cubeC_actor)
            self.cubes.append(cubeD_actor)
            self.cubes.append(cubeE_actor)
            self.cubes.append(cubeF_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_link7")
        self.cubeA_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cubeA_actor, "cubeA_handle")
        self.cubeB_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cubeB_actor, "cubeB_handle")
        self.cubeC_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cubeC_actor, "cubeC_handle")
        self.cubeD_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cubeD_actor, "cubeD_handle")
        self.cubeE_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cubeE_actor, "cubeE_handle")
        self.cubeF_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cubeE_actor, "cubeF_handle")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")
        self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(self.num_envs, self.num_props, 13)

        self.init_data()

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_link7")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_leftfinger")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_rightfinger")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        franka_local_grasp_pose = hand_pose_inv * finger_pose
        franka_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
        self.franka_local_grasp_pos = to_torch([franka_local_grasp_pose.p.x, franka_local_grasp_pose.p.y,
                                                franka_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.franka_local_grasp_rot = to_torch([franka_local_grasp_pose.r.x, franka_local_grasp_pose.r.y,
                                                franka_local_grasp_pose.r.z, franka_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        cubeA_local_grasp_pose = gymapi.Transform()
        cubeA_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        cubeA_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.cubeA_local_grasp_pos = to_torch([cubeA_local_grasp_pose.p.x, cubeA_local_grasp_pose.p.y,
                                                cubeA_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.cubeA_local_grasp_rot = to_torch([cubeA_local_grasp_pose.r.x, cubeA_local_grasp_pose.r.y,
                                                cubeA_local_grasp_pose.r.z, cubeA_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        cubeB_local_grasp_pose = gymapi.Transform()
        cubeB_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        cubeB_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.cubeB_local_grasp_pos = to_torch([cubeB_local_grasp_pose.p.x, cubeB_local_grasp_pose.p.y,
                                                cubeB_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.cubeB_local_grasp_rot = to_torch([cubeB_local_grasp_pose.r.x, cubeB_local_grasp_pose.r.y,
                                                cubeB_local_grasp_pose.r.z, cubeB_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        cubeC_local_grasp_pose = gymapi.Transform()
        cubeC_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        cubeC_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.cubeC_local_grasp_pos = to_torch([cubeC_local_grasp_pose.p.x, cubeC_local_grasp_pose.p.y,
                                                cubeC_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.cubeC_local_grasp_rot = to_torch([cubeC_local_grasp_pose.r.x, cubeC_local_grasp_pose.r.y,
                                                cubeC_local_grasp_pose.r.z, cubeC_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        cubeD_local_grasp_pose = gymapi.Transform()
        cubeD_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        cubeD_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.cubeD_local_grasp_pos = to_torch([cubeD_local_grasp_pose.p.x, cubeD_local_grasp_pose.p.y,
                                                cubeD_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.cubeD_local_grasp_rot = to_torch([cubeD_local_grasp_pose.r.x, cubeD_local_grasp_pose.r.y,
                                                cubeD_local_grasp_pose.r.z, cubeD_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        cubeE_local_grasp_pose = gymapi.Transform()
        cubeE_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        cubeE_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.cubeE_local_grasp_pos = to_torch([cubeE_local_grasp_pose.p.x, cubeE_local_grasp_pose.p.y,
                                                cubeE_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.cubeE_local_grasp_rot = to_torch([cubeE_local_grasp_pose.r.x, cubeE_local_grasp_pose.r.y,
                                                cubeE_local_grasp_pose.r.z, cubeE_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        cubeF_local_grasp_pose = gymapi.Transform()
        cubeF_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        cubeF_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.cubeF_local_grasp_pos = to_torch([cubeF_local_grasp_pose.p.x, cubeF_local_grasp_pose.p.y,
                                                cubeF_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.cubeF_local_grasp_rot = to_torch([cubeF_local_grasp_pose.r.x, cubeF_local_grasp_pose.r.y,
                                                cubeF_local_grasp_pose.r.z, cubeF_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))



        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.cubeA_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.cubeB_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.cubeC_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.cubeD_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.cubeE_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.cubeF_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.cubeA_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.cubeB_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.cubeC_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.cubeD_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.cubeE_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.cubeF_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        self.franka_grasp_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_grasp_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_grasp_rot[..., -1] = 1  # xyzw
        self.cubeA_grasp_pos = torch.zeros_like(self.cubeA_local_grasp_pos)
        self.cubeA_grasp_rot = torch.zeros_like(self.cubeA_local_grasp_rot)
        self.cubeA_grasp_rot[..., -1] = 1
        self.cubeB_grasp_pos = torch.zeros_like(self.cubeA_local_grasp_pos)
        self.cubeB_grasp_rot = torch.zeros_like(self.cubeA_local_grasp_rot)
        self.cubeB_grasp_rot[..., -1] = 1
        self.cubeC_grasp_pos = torch.zeros_like(self.cubeA_local_grasp_pos)
        self.cubeC_grasp_rot = torch.zeros_like(self.cubeA_local_grasp_rot)
        self.cubeC_grasp_rot[..., -1] = 1
        self.cubeD_grasp_pos = torch.zeros_like(self.cubeA_local_grasp_pos)
        self.cubeD_grasp_rot = torch.zeros_like(self.cubeA_local_grasp_rot)
        self.cubeD_grasp_rot[..., -1] = 1
        self.cubeE_grasp_pos = torch.zeros_like(self.cubeA_local_grasp_pos)
        self.cubeE_grasp_rot = torch.zeros_like(self.cubeA_local_grasp_rot)
        self.cubeE_grasp_rot[..., -1] = 1
        self.cubeF_grasp_pos = torch.zeros_like(self.cubeA_local_grasp_pos)
        self.cubeF_grasp_rot = torch.zeros_like(self.cubeA_local_grasp_rot)
        self.cubeF_grasp_rot[..., -1] = 1
        self.franka_lfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_rfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_lfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_rfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)

    def compute_reward(self, actions):
        self.gt_rew_buf, self.reset_buf[:], self.successes[:], self.consecutive_successes[:] = compute_success(
            self.reset_buf, self.progress_buf, self.successes, self.consecutive_successes, self.actions, self.cubeA_dof_pos,self.cubeB_dof_pos,self.cubeC_dof_pos,self.cubeD_dof_pos,self.cubeE_dof_pos,self.cubeF_dof_pos,
            self.franka_grasp_pos, self.cubeA_grasp_pos,self.cubeB_grasp_pos,self.cubeC_grasp_pos,self.cubeD_grasp_pos,self.cubeE_grasp_pos, self.franka_grasp_rot, self.cubeA_grasp_rot,self.cubeB_grasp_rot,self.cubeC_grasp_rot,self.cubeD_grasp_rot,self.cubeE_grasp_rot,self.cubeF_grasp_rot,
            self.franka_lfinger_pos, self.franka_rfinger_pos,
            self.gripper_forward_axis, self.cubeA_inward_axis,self.cubeB_inward_axis,self.cubeC_inward_axis,self.cubeD_inward_axis,self.cubeA_inward_axis,self.cubeA_inward_axis, self.gripper_up_axis, self.cubeA_up_axis,self.cubeB_up_axis,self.cubeC_up_axis,self.cubeD_up_axis,self.cubeE_up_axis,self.cubeF_up_axis,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length
        )
        self.extras['gt_reward'] = self.gt_rew_buf.mean()
        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes.mean() 

    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        cubeA_pos = self.rigid_body_states[:, self.cubeA_handle][:, 0:3]
        cubeA_rot = self.rigid_body_states[:, self.cubeA_handle][:, 3:7]
        cubeB_pos = self.rigid_body_states[:, self.cubeB_handle][:, 0:3]
        cubeB_rot = self.rigid_body_states[:, self.cubeB_handle][:, 3:7]
        cubeC_pos = self.rigid_body_states[:, self.cubeC_handle][:, 0:3]
        cubeC_rot = self.rigid_body_states[:, self.cubeC_handle][:, 3:7]
        cubeD_pos = self.rigid_body_states[:, self.cubeD_handle][:, 0:3]
        cubeD_rot = self.rigid_body_states[:, self.cubeD_handle][:, 3:7]
        cubeE_pos = self.rigid_body_states[:, self.cubeE_handle][:, 0:3]
        cubeF_rot = self.rigid_body_states[:, self.cubeF_handle][:, 3:7]

        self.franka_grasp_rot[:], self.franka_grasp_pos[:], self.cubeA_grasp_rot[:], self.cubeA_grasp_pos[:],self.cubeB_grasp_rot[:], self.cubeB_grasp_pos[:],self.cubeC_grasp_rot[:], self.cubeC_grasp_pos[:],self.cubeD_grasp_rot[:], self.cubeD_grasp_pos[:],self.cubeE_grasp_rot[:], self.cubeE_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.franka_local_grasp_rot, self.franka_local_grasp_pos,
                                     cubeA_rot, cubeA_pos, self.cubeA_local_grasp_rot, self.cubeA_local_grasp_pos,
                                     cubeB_rot, cubeB_pos, self.cubeB_local_grasp_rot, self.cubeB_local_grasp_pos,
                                     cubeC_rot, cubeC_pos, self.cubeC_local_grasp_rot, self.cubeC_local_grasp_pos,
                                     cubeD_rot, cubeD_pos, self.cubeD_local_grasp_rot, self.cubeD_local_grasp_pos,
                                     cubeE_rot, cubeE_pos, self.cubeE_local_grasp_rot, self.cubeE_local_grasp_pos,
                                     cubeF_rot, cubeF_pos, self.cubeF_local_grasp_rot, self.cubeF_local_grasp_pos
                                     )

        self.franka_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.franka_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.franka_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.franka_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        dof_pos_scaled = (2.0 * (self.franka_dof_pos - self.franka_dof_lower_limits)
                          / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)
        to_target_A = self.cubeA_grasp_pos - self.franka_grasp_pos
        to_target_B = self.cubeB_grasp_pos - self.franka_grasp_pos
        to_target_C = self.cubeC_grasp_pos - self.franka_grasp_pos
        to_target_D = self.cubeD_grasp_pos - self.franka_grasp_pos
        to_target_E = self.cubeE_grasp_pos - self.franka_grasp_pos
        to_target_F = self.cubeF_grasp_pos - self.franka_grasp_pos
        self.obs_buf = torch.cat((dof_pos_scaled, self.franka_dof_vel * self.dof_vel_scale, to_target_A,to_target_B,to_target_C,to_target_D,to_target_E,
                                  self.cubeA_dof_pos[:, 3].unsqueeze(-1), self.cubeA_dof_vel[:, 3].unsqueeze(-1),
                                  self.cubeB_dof_pos[:, 3].unsqueeze(-1), self.cubeB_dof_vel[:, 3].unsqueeze(-1),
                                  self.cubeC_dof_pos[:, 3].unsqueeze(-1), self.cubeC_dof_vel[:, 3].unsqueeze(-1),
                                  self.cubeD_dof_pos[:, 3].unsqueeze(-1), self.cubeD_dof_vel[:, 3].unsqueeze(-1),
                                  self.cubeE_dof_pos[:, 3].unsqueeze(-1), self.cubeE_dof_vel[:, 3].unsqueeze(-1),
                                  self.cubeF_dof_pos[:, 3].unsqueeze(-1), self.cubeF_dof_vel[:, 3].unsqueeze(-1)), dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        self.cubeA_dof_state[env_ids, :] = torch.zeros_like(self.cubeA_dof_state[env_ids])
        self.cubeB_dof_state[env_ids, :] = torch.zeros_like(self.cubeB_dof_state[env_ids])
        self.cubeC_dof_state[env_ids, :] = torch.zeros_like(self.cubeC_dof_state[env_ids])
        self.cubeD_dof_state[env_ids, :] = torch.zeros_like(self.cubeD_dof_state[env_ids])
        self.cubeE_dof_state[env_ids, :] = torch.zeros_like(self.cubeE_dof_state[env_ids])
        self.cubeF_dof_state[env_ids, :] = torch.zeros_like(self.cubeF_dof_state[env_ids])

        if self.num_props > 0:
            prop_indices = self.global_indices[env_ids, 2:].flatten()
            self.prop_states[env_ids] = self.default_prop_states[env_ids]
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(prop_indices), len(prop_indices))

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                px = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.cubeA_grasp_pos[i] + quat_apply(self.cubeA_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.cubeA_grasp_pos[i] + quat_apply(self.cubeA_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.cubeA_grasp_pos[i] + quat_apply(self.cubeA_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.cubeA_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.cubeB_grasp_pos[i] + quat_apply(self.cubeB_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.cubeB_grasp_pos[i] + quat_apply(self.cubeB_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.cubeB_grasp_pos[i] + quat_apply(self.cubeB_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.cubeB_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.cubeC_grasp_pos[i] + quat_apply(self.cubeC_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.cubeC_grasp_pos[i] + quat_apply(self.cubeC_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.cubeC_grasp_pos[i] + quat_apply(self.cubeC_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.cubeC_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.cubeD_grasp_pos[i] + quat_apply(self.cubeD_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.cubeD_grasp_pos[i] + quat_apply(self.cubeD_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.cubeD_grasp_pos[i] + quat_apply(self.cubeD_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.cubeD_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.cubeE_grasp_pos[i] + quat_apply(self.cubeE_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.cubeE_grasp_pos[i] + quat_apply(self.cubeE_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.cubeE_grasp_pos[i] + quat_apply(self.cubeE_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.cubeF_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.cubeF_grasp_pos[i] + quat_apply(self.cubeF_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.cubeF_grasp_pos[i] + quat_apply(self.cubeF_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.cubeF_grasp_pos[i] + quat_apply(self.cubeF_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.cubeF_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])


                px = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos,
                             cubeA_rot, cubeA_pos,cubeB_rot, cubeB_pos,cubeC_rot, cubeC_pos,cubeD_rot, cubeD_pos,cubeE_rot, cubeE_pos, cubeF_pos, 
                             cubeA_local_grasp_rot, cubeA_local_grasp_pos,cubeB_local_grasp_rot, cubeB_local_grasp_pos,cubeC_local_grasp_rot, cubeC_local_grasp_pos,cubeD_local_grasp_rot, cubeD_local_grasp_pos,cubeE_local_grasp_rot, cubeE_local_grasp_pos,cubeF_local_grasp_rot, cubeF_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos)
    global_cubeA_rot, global_cubeA_pos = tf_combine(
        cubeA_rot, cubeA_pos, cubeA_local_grasp_rot, cubeA_local_grasp_pos)
    global_cubeB_rot, global_cubeB_pos = tf_combine(
        cubeB_rot, cubeB_pos, cubeB_local_grasp_rot, cubeB_local_grasp_pos)
    global_cubeC_rot, global_cubeC_pos = tf_combine(
        cubeC_rot, cubeC_pos, cubeC_local_grasp_rot, cubeC_local_grasp_pos)
    global_cubeD_rot, global_cubeD_pos = tf_combine(
        cubeD_rot, cubeD_pos, cubeD_local_grasp_rot, cubeD_local_grasp_pos)
    global_cubeE_rot, global_cubeE_pos = tf_combine(
        cubeE_rot, cubeA_pos, cubeE_local_grasp_rot, cubeE_local_grasp_pos)
    global_cubeF_rot, global_cubeF_pos = tf_combine(
        cubeF_rot, cubeA_pos, cubeF_local_grasp_rot, cubeF_local_grasp_pos)

    return global_franka_rot, global_franka_pos, global_cubeA_rot, global_cubeA_pos,global_cubeB_rot, global_cubeB_pos,global_cubeC_rot, global_cubeC_pos,global_cubeD_rot, global_cubeD_pos,global_cubeE_rot, global_cubeE_pos




@torch.jit.script
def compute_success(
    reset_buf, progress_buf, successes, consecutive_successes, actions, cubeA_dof_pos,cubeB_dof_pos,cubeC_dof_pos,cubeD_dof_pos,cubeE_dof_pos,cubeF_dof_pos,
    franka_grasp_pos, cubeA_grasp_pos, cubeB_grasp_pos,cubeC_grasp_pos,cubeD_grasp_pos,cubeE_grasp_pos,cubeF_grasp_pos,franka_grasp_rot, cubeA_grasp_rot,cubeB_grasp_rot,cubeC_grasp_rot,cubeD_grasp_rot,cubeE_grasp_rot,cubeF_grasp_rot,
    franka_lfinger_pos, franka_rfinger_pos,
    gripper_forward_axis, cubeA_inward_axis, cubeB_inward_axis,cubeC_inward_axis,cubeD_inward_axis,cubeE_inward_axis,gripper_up_axis, cubeA_up_axis,cubeB_up_axis,cubeC_up_axis,cubeD_up_axis,cubeE_up_axis,cubeF_up_axis,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale, stack_reward_scale,lift_reward_scale,align_reward_scale
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
    cubeA_size,cubeB_size,cubeC_size,cubeD_size,cubeE_size,cubeF_size
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    target_height_B = cubeA_size[2] + cubeB_size[2] / 2.0
    target_height_D = cubeC_size[2] + cubeD_size[2] / 2.0
    target_height_F = cubeE_size[2] + cubeF_size[2] / 2.0

    # distance from hand to the cubeB
    d_b = torch.norm(cubeB_dof_pos-franka_grasp_pos, dim=-1)
    d_lf = torch.norm(cubeB_dof_pos - franka_lfinger_pos, dim=-1)
    d_rf = torch.norm(cubeB_dof_pos - franka_rfinger_pos, dim=-1)
    dist_reward_B = 1 - torch.tanh(10.0 * (d_b + d_lf + d_rf) / 3)

    # distance from hand to the cubeD
    d_d = torch.norm(cubeD_dof_pos-franka_grasp_pos, dim=-1)
    d_lf = torch.norm(cubeD_dof_pos - franka_lfinger_pos, dim=-1)
    d_rf = torch.norm(cubeD_dof_pos - franka_rfinger_pos, dim=-1)
    dist_reward_D = 1 - torch.tanh(10.0 * (d_d + d_lf + d_rf) / 3)

    # distance from hand to the cubeF
    d_f = torch.norm(cubeF_dof_pos-franka_grasp_pos, dim=-1)
    d_lf = torch.norm(cubeF_dof_pos - franka_lfinger_pos, dim=-1)
    d_rf = torch.norm(cubeF_dof_pos - franka_rfinger_pos, dim=-1)
    dist_reward_F = 1 - torch.tanh(10.0 * (d_f + d_lf + d_rf) / 3)

    dist_reward=dist_reward_B+dist_reward_D+dist_reward_F

    # reward for lifting cubeD
    cubeB_height = cubeB_dof_pos[:, 2]
    cubeB_lifted = (cubeB_height - cubeB_size[2]) > 0.04
    lift_reward_B = cubeB_lifted

    # reward for lifting cubeD
    cubeD_height = cubeD_dof_pos[:, 2]
    cubeD_lifted = (cubeD_height - cubeD_size[2]) > 0.04
    lift_reward_D = cubeD_lifted

    # reward for lifting cubeE
    cubeF_height = cubeF_dof_pos[:, 2]
    cubeF_lifted = (cubeF_height - cubeF_size[2]) > 0.04
    lift_reward_F = cubeF_lifted

    lift_reward=lift_reward_D+lift_reward_F

    # how closely aligned cubeB is to cubeA (only provided if cubeB is lifted)
    relative_b=cubeB_dof_pos-cubeA_dof_pos
    offset = torch.zeros_like(relative_b)
    offset[:, 2] = (cubeB_size[2] + cubeA_size[2]) / 2
    d_ab = torch.norm(relative_b + offset, dim=-1)
    align_reward_B = (1 - torch.tanh(10.0 * d_ab)) * cubeB_lifted

    # how closely aligned cubeD is to cubeA and cubeB (only provided if cubeD is lifted)
    relative_d=cubeD_dof_pos-cubeC_dof_pos
    offset = torch.zeros_like(relative_d)
    offset[:, 2] = (cubeC_size[2] + cubeD_size[2]) / 2
    d_cd = torch.norm(relative_d + offset, dim=-1)
    align_reward_D = (1 - torch.tanh(10.0 * d_cd)) * cubeD_lifted

    # how closely aligned cubeE is to cubeB and cubeC (only provided if cubeE is lifted)
    relative_f=cubeF_dof_pos-cubeE_dof_pos
    offset = torch.zeros_like(relative_f)
    offset[:, 2] = (cubeE_size[2] + cubeF_size[2]) / 2
    d_ef = torch.norm(relative_f + offset, dim=-1)
    align_reward_F = (1 - torch.tanh(10.0 * d_ef)) * cubeF_lifted

    align_reward=align_reward_B+align_reward_D+align_reward_E

    # Dist reward is maximum of dist and align reward
    Dist_reward = torch.max(dist_reward, align_reward)

    # final reward for stacking successfully (only if cubeB is close to target height and corresponding location, and gripper is not grasping)
    cubeB_align_cubeA = (torch.norm(relative_b[:, :2], dim=-1) < 0.02)
    cubeB_on_cubeA = torch.abs(cubeB_height - target_height_B) < 0.02
    gripper_away_from_cubeB = (d_b > 0.04)
    stack_reward_B = cubeB_align_cubeA & cubeB_on_cubeA & gripper_away_from_cubeB

    # final reward for stacking successfully (only if cubeD is close to target height and corresponding location, and gripper is not grasping)
    cubeD_align_cubeC = (torch.norm(relative_d[:, :2], dim=-1) < 0.02)
    cubeD_on_cubeC = torch.abs(cubeD_height - target_height_D) < 0.02
    gripper_away_from_cubeD = (d_d > 0.04)
    stack_reward_D = cubeD_align_cubeC & cubeD_on_cubeC & gripper_away_from_cubeD

    # final reward for stacking successfully (only if cubeE is close to target height and corresponding location, and gripper is not grasping)
    cubeF_align_cubeE = (torch.norm(relative_f[:, :2], dim=-1) < 0.02)
    cubeF_on_cubeE = torch.abs(cubef_height - target_height_f) < 0.02
    gripper_away_from_cubeF = (d_f > 0.04)
    stack_reward_F = cubeF_align_cubeE & cubeF_on_cubeE & gripper_away_from_cubeF

    stack_reward=stack_reward_B+stack_reward_D+stack_reward_F

    # Compose rewards

    # We either provide the stack reward or the align + dist reward
    rewards = torch.where(
        stack_reward,
        stack_reward_scale* stack_reward,
        dist_reward_scale * Dist_reward + lift_reward_scale * lift_reward + align_reward_scale * align_reward,
    )


    successes = torch.where(cubeA_dof_pos[:, 3] > 0.39, torch.ones_like(successes), successes)
    reset_buf = torch.where(cubeA_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    consecutive_successes = torch.where(reset_buf > 0, successes * reset_buf, consecutive_successes).mean()
    return rewards, reset_buf, successes, consecutive_successes