for i in {1..10}; do
    for storage in "s3" "scratch"; do
        mkdir "/iarai/home/ivan.svogor/storage-benchmarking/benchmark_output/repeated_long_e2e/0301_run_${i}/${storage}"
        mv "/iarai/home/ivan.svogor/storage-benchmarking/benchmark_output/repeated_long_e2e/0301_run_${i}"/*_${storage}_*/ "/iarai/home/ivan.svogor/storage-benchmarking/benchmark_output/repeated_long_e2e/0301_run_${i}/${storage}"
    done
done
