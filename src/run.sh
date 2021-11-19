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

# DataLoader
# for fetch_impl in "threaded" "asyncio"; do
#   for storage in "s3" "scratch"; do
#     for batch_size in 8 16 32 64; do
#       for num_workers in 0 2 4 8 16; do
#         for num_fetch_workers in 4 8 16 32; do
#           python3 benchmark/benchmark_dataloader.py --output_base_folder /iarai/home/ivan.svogor/git/storage-benchmarking/src/benchmark_output/dataloader-0510 \
#           --dataset "${storage}"  \
#           --num_fetch_workers "${num_fetch_workers}" \
#           --num_workers  "${num_workers}" \
#           --repeat 1 \
#           --num_batches 50 \
#           --batch_size "${batch_size}" \
#           --prefetch_factor 2 \
#           --fetch_impl "${fetch_impl}"
#         done
#       done
#     done
#   done
# done

# End2End
# for fetch_impl in "vanilla" "threaded" "asyncio" ; do
#   for storage in "s3"; do
#     for batch_size in 128 256 512; do
#       for num_workers in 0 4 16 32; do
#         python3 train/imagenet.py --output_base_folder /iarai/home/ivan.svogor/git/storage-benchmarking/src/benchmark_output/e2e1210 \
#         --dataset "${storage}" \
#         --num-fetch-workers 32 \
#         --num-workers "${num_workers}" \
#         --dataset-limit 50000 \
#         --batch-size "${batch_size}" \
#         --prefetch-factor 2 \
#         --fetch-impl "${fetch_impl}" \
#         --pin-memory 0 \
#         --accelerator dp
#       done
#     done
#   done
# done

# # for fetch_impl in "vanilla" "threaded" "asyncio" ; do
# for fetch_impl in "vanilla" "threaded" "asyncio" ; do
#   for storage in "s3" "scratch"; do
#     # for batch_size in 20 40 60 150 250 300 350; do
#     for batch_size in 350; do
#         python3 train/imagenet.py --output_base_folder /iarai/home/ivan.svogor/git/storage-benchmarking/src/benchmark_output/e2e2510 \
#         --dataset "${storage}" \
#         --num-fetch-workers 16 \
#         --num-workers 8 \
#         --dataset-limit 7000 \
#         --batch-size "${batch_size}" \
#         --prefetch-factor 16 \
#         --fetch-impl "${fetch_impl}" \
#         --pin-memory 1 \
#         # --accelerator ddp \
#         --gpus 1 \
#         --num_sanity_val_steps 0
#     done
#   done
# done

for fetch_impl in "threaded" "asyncio" "vanilla" ; do
  for storage in "s3" "scratch"; do
    for cache in 0 1; do
      for implementation in "train/imagenet_torch.py" "train/imagenet_lightning.py" ; do
          python3 "${implementation}" --output_base_folder /iarai/home/ivan.svogor/git/storage-benchmarking/src/benchmark_output/1711-cache \
          --dataset "${storage}" \
          --num-fetch-workers 16 \
          --num-workers 4 \
          --batch-pool 256 \
          --dataset-limit 3000 \
          --batch-size 64 \
          --prefetch-factor 4 \
          --fetch-impl "${fetch_impl}" \
          --use-cache "${cache}" \
          --num_sanity_val_steps 0
      done
    done
  done
done

