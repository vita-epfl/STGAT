import torch


##########################################################
###### TODO : MAKE THE DUMMY __CALL__ AS PARTH SAID ######
##########################################################

class DummyGAT:
    """
    Takes a single-pedestrian scene and makes static predictions.
    """
    def __init__(self, args):
        self.pred_len = args.pred_len

    def __call__(self, obs_traj_rel, obs_traj, seq_start_end, *args):
        out = torch.ones(self.pred_len, 1, 2)
        last_obs_pos = obs_traj[-1, 0, :]
        out[..., :] = last_obs_pos
        
        return out.cuda() 
