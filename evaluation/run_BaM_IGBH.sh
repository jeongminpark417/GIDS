CUDA_VISIBLE_DEVICES=0  python heterogeneous_train.py --dataset_size full --path /mnt/raid0/   --dataset_size full --epochs 1 --log_every 1000 --uva_graph 1 --GIDS --batch_size 1024  --data IGB --model_type rsage --num_layers 3 --fan_out '10,5,5' --cache_size $((4*1024)) --num_ssd 1   --num_ele $((550*1000*1000*1024)) --page_size 4096 --emb_size 1024 

