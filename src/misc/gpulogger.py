import threading
import subprocess
import time
import json
import logging 

class GPUSidecarLogger:
    def __init__(self, refresh_rate:float = 0.5, max_runs: int = 10) -> None:
        self.max_runs = max_runs
        self.refresh_rate = refresh_rate
        self.gpu_utilization_history = {}
        self.current_run = 0
        self.run = True

    def start(self):
        sp = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out_str = sp.communicate()
        response = out_str[0].decode("utf-8")
        gpu_usage = {}
        response = response.split("\n")
        for line in response:
            line = line.replace("%","").split(",")
            if(len(line) > 1):
                gpu_usage[line[0]] = {"gpu": line[1], 
                                      "temp": float(line[2]), 
                                      "gpu_util": float(line[3]), 
                                      "mem_util": float(line[4]),
                                     }
        logging.getLogger("gpuutil").debug(json.dumps({"gpu_data": gpu_usage, "timestamp": time.time()}))
        self.current_run += 1
        if (self.max_runs == -1 or self.max_runs > self.current_run) and self.run:
            threading.Timer(self.refresh_rate, self.start).start()

    def stop(self):
        self.run = False

if __name__ == "__main__":
    refresh_rate = 0.5
    max_runs_minutes = -1
    gpu_logger = GPUSidecarLogger(refresh_rate=refresh_rate, max_runs=max_runs_minutes)
    gpu_logger.start()
