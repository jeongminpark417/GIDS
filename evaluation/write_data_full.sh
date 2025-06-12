
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/paper/node_feat.npy  --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --pages $((1024*1024)) --n_ctrls 2  --ioffset 0
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/author/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --pages $((1024*1024)) --n_ctrls 2 --ioffset 0 --loffset $((269346174*4096))


../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/fos/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --pages $((1024*1024)) --n_ctrls 2 --loffset $((546567057*4096))  --ioffset 128
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/institute/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --pages $((1024*1024)) --n_ctrls 2 --loffset $((547280017 * 4096))  --ioffset 128

../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/journal/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --pages $((1024*1024)) --n_ctrls 2 --loffset $((546593975*4096))  --ioffset 128
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/conference/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --pages $((1024*1024)) --n_ctrls 2 --loffset $((546643027 * 4096))  --ioffset 128


