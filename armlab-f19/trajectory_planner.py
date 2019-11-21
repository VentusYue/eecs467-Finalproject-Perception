import numpy as np
import time
import math

THRESHOLD = 0.04

"""
TODO: build a trajectory generator and waypoint planner
        so it allows your state machine to iterate through
        the plan.
"""


class TrajectoryPlanner():
    def __init__(self, rexarm):
        self.idle = True
        self.rexarm = rexarm
        self.num_joints = rexarm.num_joints
        self.initial_wp = [0.0] * self.num_joints
        self.final_wp = [0.0] * self.num_joints

    def set_initial_wp(self):
        self.initial_wp = self.rexarm.get_positions

    def set_final_wp(self, waypoint):
        self.final_wp = waypoint

    def go(self):
        self.rexarm.set_positions(
             list(self.final_wp) + [self.rexarm.open_gripper()])
        time.sleep(1)
        # while True:
        #     if self.stop():
        #         break
        
        self.rexarm.set_positions(
            list(self.final_wp) + [self.rexarm.close_gripper()])
        time.sleep(1)
    def go_with_closed_gripper(self):
        # self.rexarm.set_positions(
        #      list(self.final_wp) + [self.rexarm.open_gripper()])
        # time.sleep(1)
        # while True:
        #     if self.stop():
        #         break
        
        self.rexarm.set_positions(
            list(self.final_wp) + [self.rexarm.close_gripper()])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", self.rexarm.close_gripper()) 
        time.sleep(1)

    def stop(self):
        result = True
        current_pos = self.rexarm.get_positions()
        for i in range(4):
            if np.abs(current_pos[i] - self.final_wp[i]) > THRESHOLD:
                result = False
                break
        return result
