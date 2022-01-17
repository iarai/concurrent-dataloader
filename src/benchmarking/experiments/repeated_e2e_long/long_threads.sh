# set max epochs to 100, 15000
for num_fetchers in 1 8 16; do
    for fetch_impl in "threaded" "asyncio" ; do
        for storage in "s3"; do
            for implementation in "../../experiment_src/e2e_imagenet_torch.py" "../../experiment_src/e2e_imagenet_lightning.py" ; do
                python3 "${implementation}" --output_base_folder "../../../../benchmark_output/repeated_long_e2e/fetchers_test2" \
                --dataset "${storage}" \
                --num-fetch-workers "${num_fetchers}" \
                --num-workers 4 \
                --batch-pool 256 \
                --dataset-limit 300 \
                --batch-size 128 \
                --prefetch-factor 2 \
                --fetch-impl "${fetch_impl}" \
                --use-cache 1 \
                --num_sanity_val_steps 0
            done
        done
    done
done
