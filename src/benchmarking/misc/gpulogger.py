import json
import logging
import subprocess
import threading
import time
import tempfile
import signal
import os 

class GPUSidecarLogger:
    def __init__(self, refresh_rate: float = 0.5, max_runs: int = 10) -> None:
        self.max_runs = max_runs
        self.refresh_rate = refresh_rate
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
            line = line.replace("%", "").split(",")
            if len(line) > 1:
                gpu_usage[line[0]] = {
                    "gpu": line[1],
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

class GPUSidecarLoggerMs:
    def __init__(self) -> None:
        self.out_file = tempfile.NamedTemporaryFile()
        self.cmd = "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,timestamp --format=csv,noheader --loop-ms=100"
        self.sp = None

    def start(self):
        self.sp = subprocess.Popen(
            [self.cmd],
            stdout=self.out_file,
            stderr=subprocess.PIPE,
            shell=True,
            preexec_fn=os.setsid
        )

    def stop(self):
        if self.sp is not None:
            os.killpg(os.getpgid(self.sp.pid), signal.SIGTERM)
            with open(self.out_file.name, "r") as f:
                gpu_usage = {}
                for line in f.readlines():
                    line = line.rstrip()
                    line = line.replace("%", "").split(",")
                    if len(line) > 1:
                        gpu_usage[line[0]] = {
                            "gpu": line[1],
                            "temp": float(line[2]),
                            "gpu_util": float(line[3]),
                            "mem_util": float(line[4]),
                            "timestamp": str(line[5]),
                        }
                        logging.getLogger("gpuutil").debug(json.dumps({"gpu_data": gpu_usage}))

if __name__ == "__main__":
    refresh_rate = 0.5
    max_runs_minutes = -1
    # gpu_logger = GPUSidecarLogger(refresh_rate=refresh_rate, max_runs=max_runs_minutes)
    # gpu_logger.start()
    gpu_logger = GPUSidecarLoggerMs()
    gpu_logger.start()
    # time.sleep(5)
    # gpu_logger.stop()