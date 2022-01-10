for i in {1..10}; do
  for pool_size in 1 2 3 4 5 6 7 10 15 20 30 40 50 60 80; do
    python3 "../../experiment_src/benchmark_dataset.py" \
    --output_base_folder "../../../../benchmark_output/0701-benchmark_dataset/run_scr_${i}" \
    --dataset "scratch" \
    --num_get_random_item 2000 \
    --pool_size "${pool_size}"
  done
done