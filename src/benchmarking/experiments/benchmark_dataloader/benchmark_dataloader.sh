for storage in "s3" "scratch"; do
  for num_fetch_workers in 1 2 4 8 16 32; do
    for num_workers in 1 2 4 8 16 32 64 128; do
      python3 "../../experiment_src/benchmark_dataloader.py" \
      --output_base_folder "../../../../benchmark_output/1512-benchmark_dataloader" \
      --dataset "${storage}"  --num_fetch_workers "${num_fetch_workers}" \
      --num_workers "${num_workers}" \
      --repeat 1 \
      --num_batches 40 \
      --batch_size 32 \
      --prefetch_factor 2
    done
  done
done
