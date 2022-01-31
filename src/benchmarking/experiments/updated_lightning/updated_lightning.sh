for fetch_impl in "threaded" "asyncio" "vanilla" ; do
    for storage in "s3" "scratch"; do
        for implementation in "../../experiment_src/e2e_imagenet_lightning.py" "../../experiment_src/e2e_imagenet_torch.py" ; do
            python3 "${implementation}" --output_base_folder "../../../../benchmark_output/updated_lightning" \
            --dataset "${storage}" \
            --num-fetch-workers 16 \
            --num-workers 4 \
            --batch-pool 512 \
            --dataset-limit 15000 \
            --batch-size 256 \
            --epochs 15 \
            --prefetch-factor 2 \
            --fetch-impl "${fetch_impl}" \
            --use-cache 1 \
            --num_sanity_val_steps 0
        done
    done
done