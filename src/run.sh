# num_workers=0, i.e. batches sequentiallyy§
# TODO without threadpool (num_fetch_workers=0)



# num_workers=0,1,2,4,8,16,32,64, num_fetch_workers==batch_size/2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_fetch_workers 16 --num_workers  1 --repeat 1 --num_batches 40 --batch_size 32 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_fetch_workers 16 --num_workers  2 --repeat 1 --num_batches 40 --batch_size 32 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_fetch_workers 16 --num_workers  4 --repeat 1 --num_batches 40 --batch_size 32 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_fetch_workers 16 --num_workers  8 --repeat 1 --num_batches 40 --batch_size 32 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_fetch_workers 16 --num_workers  16 --repeat 1 --num_batches 40 --batch_size 32 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_fetch_workers 16 --num_workers  32 --repeat 1 --num_batches 40 --batch_size 32 --prefetch_factor 2

python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_fetch_workers 16 --num_workers  1 --repeat 1 --num_batches 40 --batch_size 32 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_fetch_workers 16 --num_workers  2 --repeat 1 --num_batches 40 --batch_size 32 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_fetch_workers 16 --num_workers  4 --repeat 1 --num_batches 40 --batch_size 32 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_fetch_workers 16 --num_workers  8 --repeat 1 --num_batches 40 --batch_size 32 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_fetch_workers 16 --num_workers  16 --repeat 1 --num_batches 40 --batch_size 32 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_fetch_workers 16 --num_workers  32 --repeat 1 --num_batches 40 --batch_size 32 --prefetch_factor 2





# num_workers=0, varying num_fetch_workers
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_workers 0 --num_fetch_workers 1 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_workers 0 --num_fetch_workers 2 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_workers 0 --num_fetch_workers 4 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_workers 0 --num_fetch_workers 8 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_workers 0 --num_fetch_workers 16 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_workers 0 --num_fetch_workers 32 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_workers 0 --num_fetch_workers 64 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3  --num_workers 0 --num_fetch_workers 128 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2

python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_workers 0 --num_fetch_workers 1 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_workers 0 --num_fetch_workers 2 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_workers 0 --num_fetch_workers 4 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_workers 0 --num_fetch_workers 8 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_workers 0 --num_fetch_workers 16 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_workers 0 --num_fetch_workers 32 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_workers 0 --num_fetch_workers 64 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch  --num_workers 0 --num_fetch_workers 128 --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2





# dataset only

python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 0
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 1
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 2
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 3
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 4
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 5
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 6
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 7
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 10
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 15
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 20
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 30
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 40
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 50
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 60
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset s3 --num_get_random_item 2000 --pool_size 80
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 0
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 1
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 2
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 3
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 4
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 5
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 6
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 7
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 10
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 15
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 20
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 30
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 40
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 50
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 60
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset scratch --num_get_random_item 2000 --pool_size 80

rsync -Wuva ~/workspaces/storage-benchmarking/benchmark_output/ christian.eichenberger@lnx-slim-1.lan.iarai.ac.at:/iarai/work/logs/storage_benchmarking/
