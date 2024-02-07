CUDA_VISIBLE_DEVICES=0  python heterogeneous_train_baseline.py --dataset_size full --path /mnt/raid0/   --dataset_size full --epochs 1  --batch_size 1024  --data IGB --model_type rsage --num_layers 3 --fan_out '10,5,5' --emb_size 1024 

