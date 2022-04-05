import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from data.loader import data_loader
from models import TrajectoryGenerator
from utils import (
    displacement_error,
    final_displacement_error,
    l2_loss,
    int_tuple,
    relative_to_abs,
    get_dset_path,
)

## TrajNet++
import trajnetplusplustools
from data_load_utils import prepare_data
from trajnet_loader import trajnet_loader
from trajnetpp_eval_utils import \
    trajnet_batch_eval, trajnet_batch_multi_eval
from helper_models import DummyGAT 


parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

parser.add_argument("--dataset_name", default="eth_data", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--fill_missing_obs", default=1, type=int)
parser.add_argument("--keep_single_ped_scenes", default=1, type=int)

parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--noise_mix_type", default="global")

parser.add_argument(
    "--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size"
)
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma"
)
parser.add_argument(
    "--hidden-units",
    type=str,
    default="16",
    help="Hidden units in each hidden layer, splitted with comma",
)
parser.add_argument(
    "--graph_network_out_dims",
    type=int,
    default=32,
    help="dims of every node after through GAT module",
)
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)


parser.add_argument("--num_samples", default=20, type=int)


parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)

parser.add_argument("--dset_type", default="val", type=str)


parser.add_argument(
    "--resume",
    default="./model_best.pth.tar",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)

parser.add_argument(
    "--sample", type=float, default=1.0, help="Dataset ratio to sample."
)

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def get_generator(checkpoint):
    n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
    )
    n_heads = [int(x) for x in args.heads.strip().split(",")]
    model = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        alpha=args.alpha,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    return ade, fde


def evaluate(args, loader, generator):
    ade, fde, pred_col, gt_col = 0.0, 0.0, 0.0, 0.0
    topk_ade, topk_fde = 0.0, 0.0 
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch

            total_traj += len(seq_start_end)

            if obs_traj.shape[1] == 1:
                model_to_use = DummyGAT(generator, args)
            else:
                model_to_use = generator
                
            multi_traj_fake = []
            for k in range(args.num_samples):

                pred_traj_fake_rel = model_to_use(
                    obs_traj_rel, obs_traj, seq_start_end, 0, 3
                )
                pred_traj_fake_rel = pred_traj_fake_rel[-args.pred_len :]

                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                multi_traj_fake.append(pred_traj_fake.cpu().numpy().transpose(1, 0, 2))
                if k == 0:
                    s_ade, s_fde, s_pred_col, s_gt_col = \
                        trajnet_batch_eval(pred_traj_fake.cpu().numpy().transpose(1, 0, 2),
                                           pred_traj_gt.cpu().numpy().transpose(1, 0, 2),
                                           seq_start_end.cpu().numpy())
                    ade += s_ade
                    fde += s_fde
                    pred_col += s_pred_col
                    gt_col += s_gt_col
                
            s_topk_ade, s_topk_fde = trajnet_batch_multi_eval(multi_traj_fake,
                                        pred_traj_gt.cpu().numpy().transpose(1, 0, 2), 
                                        seq_start_end.cpu().numpy())
            topk_ade += s_topk_ade
            topk_fde += s_topk_fde

        ade /= total_traj
        fde /= total_traj
        pred_col /= total_traj
        gt_col /= total_traj
        topk_ade /= total_traj
        topk_fde /= total_traj
        print("Total Traj: ", total_traj)
        return ade, fde, pred_col * 100, gt_col * 100, topk_ade, topk_fde


def main(args):
    args.resume = \
        os.path.join('models', args.dataset_name, 'model_best.pth.tar')

    checkpoint = torch.load(args.resume)
    generator = get_generator(checkpoint)

    test_loader, _, _ = prepare_data(
        'datasets/' + args.dataset_name, 
        subset='/test_private/', 
        sample=args.sample
        )

    traj_test_loader = trajnet_loader(
        test_loader, 
        args, 
        drop_distant_ped=False, 
        test=True,
        keep_single_ped_scenes=args.keep_single_ped_scenes,
        fill_missing_obs=args.fill_missing_obs
        )

    ade, fde, pred_col, gt_col, topk_ade, topk_fde = \
        evaluate(args, traj_test_loader, generator)
    
    print(
        "Dataset: {}, Pred Len: {}, ADE: {:.3f}, FDE: {:.3f}, Pred: {:.3f}, GT: {:.3f}, Topk ADE: {:.3f}, Topk FDE: {:.3f}".format(
            args.dataset_name, args.pred_len, ade, fde, pred_col, gt_col, topk_ade, topk_fde
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
