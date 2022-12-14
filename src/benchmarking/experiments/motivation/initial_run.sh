# for fetch_impl in "threaded" "asyncio" "vanilla" ; do
# for fetch_impl in "vanilla" ; do
#   for storage in "scratch" "s3"; do
#     for implementation in "../../experiment_src/e2e_imagenet_torch.py" "../../experiment_src/e2e_imagenet_lightning.py" ; do
#         python3 "${implementation}" --output_base_folder "../../../../benchmark_output/motivation/rep-enh-gpu" \
#         --dataset "${storage}" \
#         --num-fetch-workers 16 \
#         --num-workers 4 \
#         --batch-pool 512 \
#         --dataset-limit 15000 \
#         --batch-size 256 \
#         --epochs 5 \
#         --prefetch-factor 4 \
#         --fetch-impl "${fetch_impl}" \
#         --use-cache 1 \
#         --num_sanity_val_steps 0
#     done
#   done
# done

# for fetch_impl in "threaded" "asyncio" "vanilla" ; do
for fetch_impl in "asyncio" ; do
  for storage in "s3"; do
    for implementation in "../../experiment_src/e2e_imagenet_torch.py"; do # "../../experiment_src/e2e_imagenet_lightning.py" ; do
        python3 "${implementation}" --output_base_folder "../../../../benchmark_output/motivation/img_count_debug" \
        --dataset "${storage}" \
        --num-fetch-workers 16 \
        --num-workers 4 \
        --batch-pool 512 \
        --dataset-limit 256 \
        --batch-size 16 \
        --epochs 1 \
        --prefetch-factor 4 \
        --fetch-impl "${fetch_impl}" \
        --use-cache 1 \
        --num_sanity_val_steps 0
    done
  done
done

