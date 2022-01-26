for batch_size in 4 8 16 32 64 128 256 512 1024 2048; do
    for i in {1..25}; do
        python3 "../../experiment_src/benchmark_tensor_loading.py" --output_base_folder "../../../../benchmark_output/batches-sleep-2/" \
        -a random_batch \
        --batch_size "${batch_size}" \
        --action_repeat 1
    done
    sleep 15 # sleep inbetween to see if there is anything related to warmup cycle
done