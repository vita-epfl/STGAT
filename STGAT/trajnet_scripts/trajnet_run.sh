MODES=3

# Train the model 
CUDA_VISIBLE_DEVICES=0 python trajnet_train.py \
    --dataset_name colfree_trajdata --obs_len 9 --num_epochs 100 \
    --fill_missing_obs 0 --keep_single_ped_scenes 0 --batch_size 1 

# Evaluate on Trajnet++ 
python -m trajnet_evaluator \
    --dataset_name colfree_trajdata --write_only --modes ${MODES} \
    --fill_missing_obs 1 --keep_single_ped_scenes 1 --batch_size 1 

    
