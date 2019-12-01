import time

class FpsCounter:
    """
    Class to count the Fps (frames per second)
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = time.time()
        return self

    def increment(self):
        self._num_occurrences += 1

    def counter(self):
        return self._num_occurrences

    def fps(self):
        elapsed_time = (time.time() - self._start_time)
        return self._num_occurrences / elapsed_time
