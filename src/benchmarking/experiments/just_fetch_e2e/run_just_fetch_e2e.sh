for fetch_impl in "threaded" "asyncio" "vanilla" ; do
  for storage in "s3" "scratch"; do
      for implementation in "../../experiment_src/e2e_imagenet_torch.py" "../../experiment_src/e2e_imagenet_lightning.py" ; do
        python3 "${implementation}" --output_base_folder "../../../../benchmark_output/1711-justfetch" \
        --dataset "${storage}" \
        --num-fetch-workers 16 \
        --num-workers 4 \
        --batch-pool 256 \
        --dataset-limit 3000 \
        --batch-size 64 \
        --prefetch-factor 4 \
        --fetch-impl "${fetch_impl}" \
        --use-cache 1 \
        --num_sanity_val_steps 0
    done
  done
done
