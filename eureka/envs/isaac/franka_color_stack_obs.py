class FrankaColorStack(VecTask):
    """Rest of the environment definition omitted."""
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
        cubeE_rot = self.rigid_body_states[:, self.cubeE_handle][:, 3:7]
        cubeF_pos = self.rigid_body_states[:, self.cubeF_handle][:, 0:3]
        cubeF_rot = self.rigid_body_states[:, self.cubeF_handle][:, 3:7]

        self.franka_grasp_rot[:], self.franka_grasp_pos[:], self.cubeA_grasp_rot[:], self.cubeA_grasp_pos[:],self.cubeB_grasp_rot[:], self.cubeB_grasp_pos[:] ,self.cubeC_grasp_rot[:], self.cubeC_grasp_pos[:] ,self.cubeD_grasp_rot[:], self.cubeD_grasp_pos[:] ,self.cubeE_grasp_rot[:], self.cubeE_grasp_pos[:] ,self.cubeF_grasp_pos[:]  = \
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
