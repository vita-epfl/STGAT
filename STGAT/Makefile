@ train:
	CUDA_VISIBLE_DEVICES=0 python trajnet_train.py --dataset_name eth_data
	CUDA_VISIBLE_DEVICES=0 python trajnet_train.py --dataset_name hotel_data
	CUDA_VISIBLE_DEVICES=0 python trajnet_train.py --dataset_name zara1_data
	CUDA_VISIBLE_DEVICES=0 python trajnet_train.py --dataset_name zara2_data
	CUDA_VISIBLE_DEVICES=0 python trajnet_train.py --dataset_name univ_data

@ test:
	CUDA_VISIBLE_DEVICES=0 python trajnet_evaluate_model.py --dataset_name eth_data --num_samples 3
	CUDA_VISIBLE_DEVICES=0 python trajnet_evaluate_model.py --dataset_name hotel_data --num_samples 3
	CUDA_VISIBLE_DEVICES=0 python trajnet_evaluate_model.py --dataset_name zara1_data --num_samples 3
	CUDA_VISIBLE_DEVICES=0 python trajnet_evaluate_model.py --dataset_name zara2_data --num_samples 3
	CUDA_VISIBLE_DEVICES=0 python trajnet_evaluate_model.py --dataset_name univ_data --num_samples 3
