for i in {1..10}; do
  for fetch_impl in "threaded" "asyncio" "vanilla" ; do
    for storage in "s3" "scratch"; do
      for implementation in "../../experiment_src/e2e_imagenet_torch.py" "../../experiment_src/e2e_imagenet_lightning.py" ; do
          python3 "${implementation}" --output_base_folder "../../../../benchmark_output/0212_run_${i}" \
          --dataset "${storage}" \
          --num-fetch-workers 16 \
          --num-workers 4 \
          --batch-pool 512 \
          --dataset-limit 15000 \
          --batch-size 256 \
          --prefetch-factor 2 \
          --fetch-impl "${fetch_impl}" \
          --use-cache 1 \
          --num_sanity_val_steps 0
      done
    done
  done
done
