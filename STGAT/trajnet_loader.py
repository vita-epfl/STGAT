import numpy as np
import torch

import trajnetplusplustools


def pre_process_test(sc_, obs_len=8):
    obs_frames = [primary_row.frame for primary_row in sc_[0]][:obs_len]
    last_frame = obs_frames[-1]
    sc_ = [[row for row in ped] for ped in sc_ if ped[0].frame <= last_frame]
    return sc_


def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask]


def trajnet_loader(data_loader, args, drop_distant_ped=False, test=False):
    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel = [], [], [], []
    loss_mask, seq_start_end = [], []
    non_linear_ped = torch.Tensor([]) # dummy
    num_batches = 0
    for batch_idx, (filename, scene_id, paths) in enumerate(data_loader):
        if test:
            paths = pre_process_test(paths, args.obs_len)
        ## Get new scene
        pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)
        if drop_distant_ped:
            pos_scene = drop_distant(pos_scene)
        # Removing Partial Tracks. Model cannot account for it !! NaNs in Loss
        full_traj = np.isfinite(pos_scene).all(axis=2).all(axis=0)
        pos_scene = pos_scene[:, full_traj]
        # Make Rel Scene
        vel_scene = np.zeros_like(pos_scene)
        vel_scene[1:] = pos_scene[1:] - pos_scene[:-1]

        # STGAT Model needs atleast 2 pedestrians per scene.
        # if sum(full_traj) > 1 and (scene_id in type3_dict[filename]):
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
