import numpy as np

# sample structure for a complex task


class task1():
    def __init__(self, fsm):
        self.fsm = fsm
        self.current_step = 0

    def operate_task(self):
        """TODO"""
        while True:
            if self.fsm.state == "idle":
                break
        self.fsm.set_current_state("moving_arm")

    def begin_task(self):
        """TODO"""
        pass
