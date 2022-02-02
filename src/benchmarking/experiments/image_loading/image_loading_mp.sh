for pool_size in 16 32; do
# for pool_size in 1 2 4 8 16 32; do
    for batch_size in 4 32 128 512 2048; do
        for i in {1..10}; do
            python3 "../../experiment_src/benchmark_tensor_loading.py" \
            --output_base_folder "../../../../benchmark_output/batches-mp-images-temp/" \
            -a random_batch_mp \
            --batch_size "${batch_size}" \
            --pool_size "${pool_size}" 
        done
        echo "DONE: ${batch_size}, ${pool_size}"
    done
done