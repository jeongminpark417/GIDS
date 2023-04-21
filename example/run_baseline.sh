
#IGB
CUDA_VISIBLE_DEVICES=0 python gids_training.py --dataset_size full --epochs 1 --log_every 1000 --uva_graph 0 --bam 0 --batch_size 4096 --window 1 --wb_size 8 --data IGB --model_type sage --num_layers 3 --fan_out '10,5,5'  > ./final_result/baseline_IGB.txt

#OGB
CUDA_VISIBLE_DEVICES=0 python gids_training.py --dataset_size full --epochs 1 --log_every 1000 --uva_graph 0 --bam 0 --batch_size 4096 --data OGB --model_type gat --num_layers 3 --fan_out '10,5,5' --path /mnt/nvme15/ogbn_papers100M/raw/ --num_classes 172 > ./final_result/baseline_OGB.txt

#MAG
CUDA_VISIBLE_DEVICES=0 python gids_training_hetero.py --dataset_size full --epochs 1  --log_every 1000 --model_type rsage --batch_size 4096 --num_layers 3 --fan_out '10,5,5' --in_memory 0 --path /mnt/nvme15/mag240m_kddcup2021/ --data OGB --num_classes 153 --bam 0 --uva_graph 0 > ./final_result/baseline_MAG.txt

#IGBH
CUDA_VISIBLE_DEVICES=0 python gids_training_hetero.py --dataset_size full --epochs 1  --log_every 1000 --model_type rsage --batch_size 4096 --num_layers 3 --fan_out '10,5,5' --path /mnt/raid0/ --in_memory 0 --bam 0 --uva_graph 0 >  ./final_result/baseline_IGBH.txt




