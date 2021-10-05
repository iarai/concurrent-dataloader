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
#           python3 benchmark/benchmark_dataloader.py --output_base_folder /iarai/home/ivan.svogor/git/storage-benchmarking/src/benchmark_output/dataloader \
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
for pin_memory in 0 1; do
  for fetch_impl in "threaded" "asyncio"; do
    for storage in "s3" "scratch"; do
      for batch_size in 16 32 64; do
        for num_workers in 0 4 8 16; do
          for num_fetch_workers in 4 8 16 32; do
            python3 train/imagenet.py --output_base_folder /iarai/home/ivan.svogor/git/storage-benchmarking/src/benchmark_output/e2e \
            --dataset "${storage}" \
            --num-fetch-workers "${num_fetch_workers}" \
            --num-workers "${num_workers}" \
            --dataset-limit 1024 \
            --batch-size "${batch_size}" \
            --prefetch-factor 2 \
            --fetch-impl "${fetch_impl}" \
            --pin-memory "${pin_memory}" \
            --accelerator dp
          done
        done
      done
    done
  done
done

