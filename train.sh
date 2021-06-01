python train_and_prune.py --dataset_path='/home/user/datasets/qvga_dataset' \
						  --logdir='log' \
						  --val_set_size=10 \
						  --validation_freq=10 \
						  --gray \
						  --early_pooling \
						  --filterdepths='32,32,32,64,64,128,128' \
						  --sparsity='0.25,0.25,0.35,0.35,0.45,0.45' \
						  --epochs='60,5,50' \
						  --gpuid=0 \
						  --configfile='squeezedet.config' \
						  --runpruning \
						  # --usecov3 \
						  # --usedefaultvalset \
						  # --freeze_landmark='fire1' \
						  # --init=<path to weights>