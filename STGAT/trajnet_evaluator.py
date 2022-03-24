import os
import argparse
import pickle

from joblib import Parallel, delayed
import scipy
import torch
from tqdm import tqdm
import trajnetplusplustools
import numpy as np

from evaluator.trajnet_evaluator import trajnet_evaluate
from evaluator.write_utils import load_test_datasets, preprocess_test, write_predictions

# STGAT
##################
from models import TrajectoryGenerator
from utils import int_tuple
# import ...
##################

def predict_scene(model, clusters, paths, args):
    pass


def load_predictor(args):
    checkpoint = torch.load(args.checkpoint)
    
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


def get_predictions(args):
    """
    Get model predictions for each test scene and write the predictions 
    in appropriate folders.
    """
    # List of .json file inside the args.path 
    # (waiting to be predicted by the testing model)
    datasets = sorted([
        f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) \
        if not f.startswith('.') and f.endswith('.ndjson')
        ])

    # Extract Model names from arguments and create its own folder 
    # in 'test_pred' for storing predictions
    # WARNING: If Model predictions already exist from previous run, 
    # this process SKIPS WRITING
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')
        model_name = model_name + '_modes' + str(args.modes)

        ## Check if model predictions already exist
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        if not os.path.exists(args.path + model_name):
            os.makedirs(args.path + model_name)
        else:
            print(f'Predictions corresponding to {model_name} already exist.')
            print('Loading the saved predictions')
            continue

        print("Model Name: ", model_name)
        model, clusters = load_predictor(args)
        goal_flag = False

        # Iterate over test datasets
        for dataset in datasets:
            # Load dataset
            dataset_name, scenes, scene_goals = \
                load_test_datasets(dataset, goal_flag, args)

            # Get all predictions in parallel. Faster!
            scenes = tqdm(scenes)
            pred_list = Parallel(n_jobs=1)(
                delayed(predict_scene)(model, clusters, paths, args)
                for (_, _, paths), scene_goal in zip(scenes, scene_goals)
                )
            
            # Write all predictions
            write_predictions(pred_list, scenes, model_name, dataset_name, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", default="eth_data", type=str
        )
    parser.add_argument(
        '--obs_len', default=9, type=int, help='observation length'
        )
    parser.add_argument(
        '--pred_len', default=12, type=int, help='prediction length'
        )
    parser.add_argument(
        '--write_only', action='store_true', help='disable writing new files'
        )
    parser.add_argument(
        '--disable-collision', action='store_true', help='disable collision metrics'
        )
    parser.add_argument(
        '--labels', required=False, nargs='+', help='labels of models'
        )
    parser.add_argument(
        '--normalize_scene', action='store_true', help='augment scenes'
        )
    parser.add_argument(
        '--modes', default=1, type=int, help='number of modes to predict'
        )

    # STGAT
    parser.add_argument(
        "--output", default="models", type=str, metavar="PATH",
        help="path to best checkpoint",
        )
    parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
    parser.add_argument("--noise_type", default="gaussian")

    parser.add_argument(
        "--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size"
        )
    parser.add_argument(
        "--traj_lstm_hidden_size", default=32, type=int
        )
    parser.add_argument(
        "--graph_network_out_dims", type=int, default=32,
        help="dims of every node after through GAT module",
        )
    parser.add_argument(
        "--graph_lstm_hidden_size", default=32, type=int
        )
    parser.add_argument(
        "--heads", type=str, default="4,1", 
        help="Heads in each layer, splitted with comma"
        )
    parser.add_argument(
        "--hidden-units", type=str, default="16",
        help="Hidden units in each hidden layer, splitted with comma",
        )
    parser.add_argument(
        "--dropout", type=float, default=0, 
        help="Dropout rate (1 - keep probability)."
        )
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
        )
    args = parser.parse_args()

    scipy.seterr('ignore')

    args.checkpoint = \
        os.path.join(args.output, args.dataset_name, 'model_best.pth.tar')
    args.path = os.path.join('datasets', args.dataset_name, 'test_pred')

    # Writes to Test_pred
    # Does NOT overwrite existing predictions if they already exist ###
    
    #######################
    # DEBUGGING
    # get_predictions(args)
    print(args.checkpoint)
    model = load_predictor(args)
    print(model)

    # if args.write_only: # For submission to AICrowd.
    #     print("Predictions written in test_pred folder")
    #     exit()

    # ## Evaluate using TrajNet++ evaluator
    # trajnet_evaluate(args)
    #######################


if __name__ == '__main__':
    main()


