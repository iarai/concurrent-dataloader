BATCH_SIZE=50
DATASOURCE="s3"

# async
echo "Starting with async"
for i in {1..5}
do
	python main.py -a wip -args $BATCH_SIZE 2 async > benchmark_output/${DATASOURCE}/1_$((2 ** $i))_${BATCH_SIZE}_async.txt
done
echo "Async done"

# sync
echo "Starting with sync"
for i in {1..5}
do
	python main.py -a wip -args $BATCH_SIZE 2 async > benchmark_output/${DATASOURCE}/1_$((2 ** $i))_${BATCH_SIZE}_sync.txt
done
echo "Sync done"
