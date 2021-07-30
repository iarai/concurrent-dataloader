from misc.time_helper import TimeHelper


class ActionPlayer:
    def __init__(self):
        self.stopwatch = TimeHelper()

    def reset(self):
        self.stopwatch.reset()

    def benchmark(self, action_name, action, repeat):
        for i in range(repeat):
            self.stopwatch.record(action_name + "_" + str(i))
            action()
            self.stopwatch.record(action_name + "_" + str(i))
        self.stopwatch.get_results(action_name, repeat)
