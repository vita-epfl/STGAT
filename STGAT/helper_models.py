import torch


class DummyGAT:
    """
    Takes a single-pedestrian scene and makes static predictions.
    """
    def __init__(self, model, args):
        self.model = model
        self.obs_len = args.obs_len

    def __call__(self, obs_traj_rel, obs_traj, seq_start_end, *args):
        # Initializing a static dummy pedestrian somewhere far away in the frame
        dummy_coords = torch.tensor([-1000., -1000.]) 
        zero_vels = torch.tensor([0., 0.])

        # Initialize with ones
        obs_traj_padded = torch.ones(self.obs_len, 2, 2).cuda()
        obs_traj_rel_padded = torch.ones(self.obs_len, 2, 2).cuda()
        seq_start_end_padded = torch.tensor([[0, 2]]).cuda()

        # Keep the primary pedestrian
        obs_traj_padded[:, 0, :] = obs_traj[:, 0, :]
        obs_traj_rel_padded[:, 0, :] = obs_traj_rel[:, 0, :]

        # Add the static dummy pedestrian
        obs_traj_padded[:, 1, :] = dummy_coords
        obs_traj_rel_padded[:, 1, :] = zero_vels

        # Compute model predictions
        pred_traj_fake_rel = self.model(
            obs_traj_rel_padded, obs_traj_padded, seq_start_end_padded, 0, 3
            )

        # Return only the primary pedestrian
        return torch.unsqueeze(pred_traj_fake_rel[:, 0, :], 1)
