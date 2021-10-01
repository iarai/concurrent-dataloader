## num_workers=0, i.e. batches sequentiallyy§
## TODO without threadpool (num_fetch_workers=0)
#
## num_workers=0,1,2,4,8,16,32,64, batch_size=32
#for storage in "s3" "scratch"; do
#  for num_fetch_workers in 1 2 4 8 16 32; do
#    for num_workers in 1 2 4 8 16 32 64 128; do
#      python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset "${storage}"  --num_fetch_workers "${num_fetch_workers}" --num_workers  "${num_workers}" --repeat 1 --num_batches 40 --batch_size 32 --prefetch_factor 2
#    done
#  done
#done
#
#
#
## num_workers=0, varying num_fetch_workers
#for storage in "s3" "scratch"; do
#  for num_fetch_workers in 1 2 4 8 16 32 64 128; do
#    python -m benchmark.benchmark_dataloader --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset "${storage}"  --num_workers 0 --num_fetch_workers "${num_fetch_workers}" --repeat 1 --num_batches 40 --batch_size 128 --prefetch_factor 2
#  done
#done
#
#
#
#
## dataset only
#for storage in "s3" "scratch"; do
#  for pool_size in 1 2 3 4 5 6 7 10 15 20 30 40 50 60 80; do
#    python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output --dataset "${storage}" --num_get_random_item 2000 --pool_size "${pool_size}"
#  done
#done
#
#
#rsync -Wuva ~/workspaces/storage-benchmarking/benchmark_output/ christian.eichenberger@lnx-slim-1.lan.iarai.ac.at:/iarai/work/logs/storage_benchmarking/

for batch_size in 8 16 32 64; do
  for num_workers in 0 2 4 8 16; do
    for num_fetch_workers in 4 8 16 32; do
      python3 benchmark/benchmark_dataloader.py --output_base_folder /home/ivan/Documents/git/storage-benchmarking/src/benchmark_output \
      --dataset s3  \
      --num_fetch_workers "${num_fetch_workers}" \
      --num_workers  "${num_workers}" \
      --repeat 1 \
      --num_batches 10 \
      --batch_size "${batch_size}" \
      --prefetch_factor 2
    done
  done
done
