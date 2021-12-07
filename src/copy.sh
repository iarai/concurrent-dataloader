# for i in {1..10}; do
#     for storage in "s3" "scratch"; do
#         mkdir "/iarai/home/ivan.svogor/git/storage-benchmarking/src/benchmark_output/0212_run_${i}/${storage}"
#         mv "/iarai/home/ivan.svogor/git/storage-benchmarking/src/benchmark_output/0212_run_${i}"/*_${storage}_*/ "/iarai/home/ivan.svogor/git/storage-benchmarking/src/benchmark_output/0212_run_${i}/${storage}"
#     done
# done

for i in {1..10}; do
    for storage in "s3" "scratch"; do
        mkdir "/iarai/home/ivan.svogor/git/storage-benchmarking/src/benchmark_output/0212_run_${i}/${storage}"
        mv "/iarai/home/ivan.svogor/git/storage-benchmarking/src/benchmark_output/0212_run_${i}"/*_${storage}_*/ "/iarai/home/ivan.svogor/git/storage-benchmarking/src/benchmark_output/0212_run_${i}/${storage}"
    done
done