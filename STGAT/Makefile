@ train:
	CUDA_VISIBLE_DEVICES=0 python trajnet_train.py --dataset_name colfree_trajdata --obs_len 9 --batch_size 1

@ test:
	CUDA_VISIBLE_DEVICES=0 python trajnet_evaluate_model.py --dataset_name colfree_trajdata --num_samples 3 --obs_len 9 --batch_size 1 

@ test_trajnet:  # For submission to AICrowd
	python -m trajnet_evaluator --dataset_name colfree_trajdata --write_only
