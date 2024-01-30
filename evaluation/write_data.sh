
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/nvme17/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1

../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/author/node_feat.npy  --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --loffset $((269346174*4096))

../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/fos/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --loffset $((546567057*4096)) 


../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/institute/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --loffset $((547280017 * 4096)) 


