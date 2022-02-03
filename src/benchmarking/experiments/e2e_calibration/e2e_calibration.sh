# set max epochs to 100, 15000
for epochs in 5000 10000 20000; do
  for fetch_impl in "threaded" "asyncio" "vanilla" ; do
    for implementation in "../../experiment_src/e2e_imagenet_torch.py" "../../experiment_src/e2e_imagenet_lightning.py" ; do
        python3 "${implementation}" --output_base_folder "../../../../benchmark_output/e2e_calibration/" \
        --dataset "s3" \
        --num-fetch-workers 16 \
        --num-workers 4 \
        --batch-pool 512 \
        --dataset-limit 15000 \
        --batch-size 256 \
        --prefetch-factor 2 \
        --epochs "${epochs}" \
        --fetch-impl "${fetch_impl}" \
        --use-cache 0 \
        --num_sanity_val_steps 0
    done
  done
done
