import os
# from io import BytesIO
from pathlib import Path
# import torch
# import torchvision.datasets as datasets
# from torchvision.datasets import imagenet
# import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
import time 
from collections import defaultdict
import numpy as np
import random
import pickle

IMAGENET_PATH_SCRATCH = "/scratch/imagenet"
IMAGENET_PATH_GLUSTER = "/scratch/imagenet"


class ScratchDataLoader(Dataset):
    def __init__(
        self,
        imagenet_path: str,
        mode,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
    ):
        self.mode = mode
        self.image_paths = None
        assert mode in ["train", "val"], mode        
        self.__imagenet_path = Path(os.path.join(imagenet_path, mode))

    def index_all(self):
        self.image_paths = list(self.__imagenet_path.rglob(f"**/*.JPEG"))   

    
    def save_index(self):
        with open("index.bin", "wb") as file:
            pickle.dump(self.image_paths, file)

    def load_index(self):
        with open("index.bin", "rb") as file:
            self.image_paths = pickle.load(file)
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        return image

    def __len__(self):
        return len(self.image_paths)


class TimeHelper():
    def __init__(self,):
        self.recordings = defaultdict(list)

    def record(self, name: str):
        self.recordings[name].append(time.time())

    def get_results(self, name: str, dataset_length: int, summary_only=True):
        diffs = []
        for i in stopwatch.recordings:
            if name in i:
                diff = stopwatch.recordings[i][1] - stopwatch.recordings[i][0]
                diffs.append(diff)
                if not summary_only:
                    print(f"{i} {diff} s {dataset_length/diff} files/s")
        mean_indexing_time = np.mean(np.array(diffs))
        print(f"{name} Mean diff {mean_indexing_time}, {dataset_length/mean_indexing_time} files/s")


if __name__ == "__main__":
    print("Hi scratch data loader!")
    imagenet_sdl = ScratchDataLoader(IMAGENET_PATH_GLUSTER, "train", 4, False, 1, False)
    stopwatch = TimeHelper()

    

    # index images
    for i in range(30):
        stopwatch.record("indexing_"+str(i))
        imagenet_sdl.index_all()
        stopwatch.record("indexing_"+str(i))  
    
    print(f"Dataset size: {imagenet_sdl.__len__()}")
    stopwatch.get_results("indexing", imagenet_sdl.__len__())


    # load random images
    for i in range(500):
        rn = random.randint(0, imagenet_sdl.__len__())
        stopwatch.record("loading_"+str(i)+"_"+str(rn))
        image = imagenet_sdl.__getitem__(rn)
        stopwatch.record("loading_"+str(i)+"_"+str(rn))
    stopwatch.get_results("loading", 500)

    # write index to file
    for i in range(100):
        stopwatch.record("storeind_"+str(i))
        imagenet_sdl.save_index()
        stopwatch.record("storeind_"+str(i))
    stopwatch.get_results("storeind", 100)

    # read index from file 
    for i in range(100):
        stopwatch.record("loadind_"+str(i))
        imagenet_sdl.load_index()
        stopwatch.record("loadind_"+str(i))
    stopwatch.get_results("loadind", 100)

    print("Recreating DataLoader")
    imagenet_sdl = ScratchDataLoader(IMAGENET_PATH_GLUSTER, "train", 4, False, 1, False)
    imagenet_sdl.load_index()
    # load random images
    for i in range(500):
        rn = random.randint(0, imagenet_sdl.__len__())
        stopwatch.record("fromfileindex_"+str(i)+"_"+str(rn))
        image = imagenet_sdl.__getitem__(rn)
        stopwatch.record("fromfileindex_"+str(i)+"_"+str(rn))
    stopwatch.get_results("fromfileindex_", 500)