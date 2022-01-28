import numpy as np
import torch

import trajnetplusplustools


def trajnet_loader(data_loader, args):
    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel = [], [], [], []
    loss_mask, seq_start_end = [], []
    non_linear_ped = torch.Tensor([]) # dummy
    num_batches = 0
    for batch_idx, (filename, scene_id, paths) in enumerate(data_loader):
        ## Get new scene
        pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)
        # Removing Partial Tracks. Model cannot account for it !! NaNs in Loss
        full_traj = np.isfinite(pos_scene).all(axis=2).all(axis=0)
        pos_scene = pos_scene[:, full_traj]
        # Make Rel Scene
        vel_scene = np.zeros_like(pos_scene)
        vel_scene[1:] = pos_scene[1:] - pos_scene[:-1]

        # STGAT Model needs atleast 2 pedestrians per scene.
        if sum(full_traj) > 1:
            # Get Obs, Preds attributes
            obs_traj.append(torch.Tensor(pos_scene[:args.obs_len]))
            pred_traj_gt.append(torch.Tensor(pos_scene[-args.pred_len:]))
            obs_traj_rel.append(torch.Tensor(vel_scene[:args.obs_len]))
            pred_traj_gt_rel.append(torch.Tensor(vel_scene[-args.pred_len:]))

            # Get Seq Delimiter and Dummy Loss Mask
            seq_start_end.append(pos_scene.shape[1])
            curr_mask = torch.ones((pos_scene.shape[0], pos_scene.shape[1]))
            loss_mask.append(curr_mask)
            num_batches += 1

        if num_batches % args.batch_size != 0 and (batch_idx + 1) != len(data_loader):
            continue
        
        if len(obs_traj):
            obs_traj = torch.cat(obs_traj, dim=1).cuda()
            pred_traj_gt = torch.cat(pred_traj_gt, dim=1).cuda()
            obs_traj_rel = torch.cat(obs_traj_rel, dim=1).cuda()
            pred_traj_gt_rel = torch.cat(pred_traj_gt_rel, dim=1).cuda()
            loss_mask = torch.cat(loss_mask, dim=1).cuda().permute(1, 0)
            seq_start_end = [0] + seq_start_end
            seq_start_end = torch.LongTensor(np.array(seq_start_end).cumsum())
            seq_start_end = torch.stack((seq_start_end[:-1], seq_start_end[1:]), dim=1)
            yield (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                non_linear_ped, loss_mask, seq_start_end)
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel = [], [], [], []
            loss_mask, seq_start_end = [], []
