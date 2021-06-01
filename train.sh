python train_and_prune.py --dataset_path='/home/user/datasets/vga-dataset' \
						  --logdir='logs' \
						  --val_set_size=50 \
						  --validation_freq=5 \
						  --gray \
						  --early_pooling \
						  --filterdepths='32,32,64,64,88,128' \
						  --sparsity='0.25,0.25,0.35,0.35,0.45' \
						  --epochs='80,5,50' \
						  --gpuid=0 \
						  --configfile='squeezedet.config' \
						  # --runpruning \
						  # --usecov3 \
						  # --usedefaultvalset \
						  # --freeze_landmark='fire1' \
						  # --init=<check point path>




