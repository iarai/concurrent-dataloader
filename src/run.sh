# async
echo "Starting with async"
python main.py -a wip -args 50 2 async > output/1_2_50_async.txt
python main.py -a wip -args 50 4 async > output/2_4_50_async.txt
python main.py -a wip -args 50 8 async > output/3_8_50_async.txt
python main.py -a wip -args 50 16 async > output/4_16_50_async.txt
python main.py -a wip -args 50 32 async > output/5_32_50_async.txt
echo "Async done"
# sync
echo "Starting with sync"
python main.py -a wip -args 50 4 sync > output/1_2_50_sync.txt
python main.py -a wip -args 50 2 sync > output/2_4_50_sync.txt
python main.py -a wip -args 50 8 sync > output/3_8_50_sync.txt
python main.py -a wip -args 50 16 sync > output/4_16_50_sync.txt
python main.py -a wip -args 50 32 sync > output/5_32_50_sync.txt
echo "Sync done"