import time
import numpy as np
import cv2

import lcm
import os
os.sys.path.append('lcmtypes/')
# Messages used for communication with Mbot programs.
# NOTE: Add your customized lcm messages as needed.
from lcmtypes import pose_xyt_t
from lcmtypes import occupancy_grid_t
from lcmtypes import mbot_status_t
from lcmtypes import mbot_command_t
from lcmtypes import robot_path_t
D2R = 3.141592 / 180.0
R2D = 180.0 / 3.141592
CUBE_LEN = 0.0254

"""
NOTE: Add states and state functions to this class
        to implement all of the required logics
"""


class StateMachine():
    def __init__(self, rexarm, planner):
        self.rexarm = rexarm
        self.tp = planner
        self.tags = []
        self.status_message = "State: Idle"
        self.current_state = "idle"

        self.lc = lcm.LCM()
        lcmSLAMPoseSub = self.lc.subscribe(
            "SLAM_POSE", self.slampose_feedback_handler)
        lcmSLAMPoseSub.set_queue_capacity(1)

        lcmMbotStatusSub = self.lc.subscribe(
            "MBOT_STATUS", self.mbotstatus_feedback_handler)

        # NOTE: Add more variables here, such as RBG/HSV image here.
        self.current_step = 0
        self.path = []
        self.start_time = 0
        self.duration = 0
        self.slam_pose = None
        self.current_goalPose = None

        self.extrinsic_inv = self.get_extrinsicInv()
        self.extrinsic = self.get_extrinsic()

        self.focal_length = 300

    def set_current_state(self, state):
        self.current_state = state

    """ This function is run continuously in a thread"""

    def run(self):
        if(self.current_state == "manual"):
            self.manual()

        if(self.current_state == "idle"):
            self.idle()

        if(self.current_state == "estop"):
            self.estop()

        if(self.current_state == "calibrate"):
            self.calibrate()

        if(self.current_state == "moving_arm"):
            self.moving_arm()
        if(self.current_state == "reach_block"):
            self.moving_arm_with_closed_gripper()
        if(self.current_state == "moving_mbot"):
            self.moving_mbot()

        if self.current_state == "arm_return_home":
            self.arm_return_home()

        if self.current_state == "swipe":
            self.swipe_arm()
        if self.current_state == "put_back_block":
            self.put_back_block()
        if self.current_state == "moving_back":
            self.moving_back()
        if self.current_state == "moving_to_block_side":
            self.moving_to_block_side()
        self.get_mbot_feedback()
        self.rexarm.get_feedback()

    """Functions run for each state"""

    def manual(self):
        self.status_message = "State: Manual - Use sliders to control arm"
        self.rexarm.send_commands()

    def idle(self):
        self.status_message = "State: Idle - Waiting for input"

    def estop(self):
        self.status_message = "EMERGENCY STOP - Check Rexarm and restart program"
        self.rexarm.disable_torque()

    def calibrate(self):
        """
        Perform camera calibration here
        Use appropriate intrinsicMatrix parameters
        Change the arm frame 3d coordinate based on the point that you choose.
        Store the extrinsic matrix and load it on start.
        """
        # NOTE: ----------------- FIRST REASONABLE DATA -----------------
        #   WORKS FINE ONLY WITHIN (x = 15cm, y = +/-5cm)
        # >>>>>>>>>>>>>> Object Points:
        # [[0.15    0.0127  0.0381]
        #  [0.15 - 0.0127  0.0381]
        #  [0.15    0.0127  0.0127]
        #  [0.15 - 0.0127  0.0127]]
        # >>>>>>>>>>>>>> Extrinsic:
        # [[-8.27275930e-02 - 9.96571841e-01 - 8.42851409e-04  5.19003157e-03]
        #  [-2.70938513e-01  2.33050726e-02 - 9.62314500e-01  6.67610085e-02]
        #  [9.59035175e-01 - 7.93816013e-02 - 2.71937665e-01 - 7.15535861e-02]
        #  [0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
        # >>>>>>>>>>>>>> poses in arm frame:
        # [[0.14508789  0.01127584  0.03811856  1.]
        #  [0.14280142 - 0.01156055  0.03902669  1.]
        #  [0.14290378  0.01151223  0.01673917  1.]
        #  [0.14086536 - 0.01147591  0.01746459  1.]]
        # ----------------------------------------------------------------

        intrinsicMatrix = np.array([[self.focal_length, 0.0, 320],
                                    [0.0, self.focal_length, 240],
                                    [0.0, 0.0, 1.0]])
        distortionCoef = np.array([2.82171790e-01, -1.55585435e+00,
                                   -1.30308250e-03, -2.73407033e-04, 2.62178923e+00])
        # 3D coordinates of the center of AprilTags in the arm frame in meters.
        objectPoints = np.array([[0.15, 0.0254 / 2, 3 * 0.0254 / 2],
                                 [0.15, 0.0254 / 2, 0.0254 / 2],
                                 [0.20, -0.0254 / 2, 3 * 0.0254 / 2],
                                 [0.20, -0.0254 / 2, 0.0254 / 2]])
        # 1.25830477 rad
        # Use the center of the tags as image points. Make sure they correspond to the 3D points.
        imagePoints = np.array([tag.center for tag in self.tags])
        print(">>>>>>>>>>>>>> tage pose")
        print(imagePoints)
        print(">>>>>>>>>>>>>> Object Points: ")
        print(objectPoints)

        ##########################################################
        # NOTE: solvePnP() for calculating the extrinsic
        #   DEPRECATED DUE TO INACCURACY
        # _, rvec, tvec = cv2.solvePnP(
        #     objectPoints, imagePoints, intrinsicMatrix, None)
        # rotation_matrix, _ = cv2.Rodrigues(rvec)

        # extrinsic = np.zeros((4, 4))

        # extrinsic[0:3, 0:3] = rotation_matrix
        # extrinsic[0:3, 3] = np.squeeze(tvec)
        # extrinsic[3, 3] = 1
        # extrinsic_inv = np.linalg.inv(extrinsic)
        ##########################################################
        # NOTE: hand-calculated, hard-coded extrinsic
        extrinsic = self.extrinsic
        extrinsic_inv = self.extrinsic_inv
        ##########################################################

        print(">>>>>>>>>>>>>> Extrinsic: ")
        print(extrinsic)
        poses_camera = np.array([tag.pose_t for tag in self.tags])
        poses_arm = []
        for pose in poses_camera:
            pose_homo = np.append(pose, [1])
            pose_arm = extrinsic_inv @ pose_homo
            poses_arm.append(pose_arm)
        poses_arm = np.array(poses_arm)
        print(">>>>>>>>>>>>>> ARM POSE: ")
        print(poses_arm)

        # Implement the function that transform pose_t of the tag to the arm's
        # frame of reference.

        self.status_message = "Calibration - Completed Calibration"
        time.sleep(1)

        # Set the next state to be idle
        self.set_current_state("idle")

    def swipe_arm(self):
        """ Implement this function"""
        # self.rexarm.send_commands()
        if self.tags:
            self.tp.set_initial_wp()

            tagPose1 = self.tags[0].pose_t

            #################### FOR DEBUG ####################
            print(">>>>>>>>>>>>>>>> TAG1 POSE:")
            print(self.extrinsic_inv @
                  np.vstack((tagPose1, np.array([1]))))
            ###################################################

            z = self.tags[0].pose_R[:, -1].reshape(3, 1)
            y = self.tags[0].pose_R[:, 1].reshape(3, 1)
            if y[0, 0] < 0:
                y *= -1
            tagPose1 += (CUBE_LEN) * z
            tagPose1 -= 0.01 * y

            homo_coor1 = self.extrinsic_inv @\
                np.vstack((tagPose1, np.array([1])))
            homo_coor1 = homo_coor1 / homo_coor1[-1]

            tagPose2 = self.tags[1].pose_t

            #################### FOR DEBUG ####################
            print(">>>>>>>>>>>>>>>> TAG2 POSE:")
            print(self.extrinsic_inv @
                  np.vstack((tagPose2, np.array([1]))))
            ###################################################

            z = self.tags[1].pose_R[:, -1].reshape(3, 1)
            y = self.tags[1].pose_R[:, 1].reshape(3, 1)
            if y[0, 0] < 0:
                y *= -1
            tagPose2 += (CUBE_LEN) * z
            tagPose2 -= 0.01 * y

            homo_coor2 = self.extrinsic_inv @\
                np.vstack((tagPose2, np.array([1])))
            homo_coor2 = homo_coor2 / homo_coor2[-1]

            homo_coor = homo_coor1

            #################### FOR DEBUG ####################
            print(">>>>>>>>>>>>>>>> HOMO COOR:")
            print(homo_coor)
            ###################################################

            if (homo_coor[1][0] > -0.01):
                # add offsets to the tag to solve for a perfect desired
                #   point for Rexarm
                if(homo_coor[1][0] < 0.02):
                    final_wp = self.rexarm.rexarm_IK(
                        [homo_coor[0][0] + 0.005, homo_coor[1][0] + 0.008, homo_coor[2][0], -90])
                else:
                    final_wp = self.rexarm.rexarm_IK(
                        [homo_coor[0][0] + 0.005, homo_coor[1][0] + 0.012, homo_coor[2][0], -90])
            else:
                final_wp = self.rexarm.rexarm_IK(
                    [homo_coor[0][0] + 0.015, homo_coor[1][0] + 0.002, homo_coor[2][0], -90])

            try:
                self.tp.set_final_wp(final_wp)
                self.tp.go()
            except TypeError:
                # not finding final_wp returns None,
                # which raises TypeError when indexing
                pass

        self.set_current_state("idle")

    def moving_arm(self):
        """ Implement this function"""
        # Rexarm reset to home position
        self.rexarm.set_positions([0.0, 0.0, 0.0, 0.0, 0.15])
        time.sleep(1.0)
        while not self.tags:
            time.sleep(1.0)
            print("not seeing any tags...")
        if self.tags:
            self.tp.set_initial_wp()
            tagPose = self.tags[0].pose_t

            #################### FOR DEBUG ####################
            print(">>>>>>>>>>>>>>>> TAG POSE:")
            print(self.extrinsic_inv @
                  np.vstack((tagPose, np.array([1]))))
            ###################################################

            z = self.tags[0].pose_R[:, -1].reshape(3, 1)
            y = self.tags[0].pose_R[:, 1].reshape(3, 1)
            if y[0, 0] < 0:
                y *= -1
            tagPose += (CUBE_LEN) * z
            tagPose -= 0.01 * y

            homo_coor = self.extrinsic_inv @\
                np.vstack((tagPose, np.array([1])))
            homo_coor = homo_coor / homo_coor[-1]
            #################### FOR DEBUG ####################
            print(">>>>>>>>>>>>>>>> HOMO COOR:")
            print(homo_coor)
            ###################################################
            # homo_coor[2][0] += 0.05
            if (homo_coor[1][0] > -0.01):
                # add offsets to the tag to solve for a perfect desired
                #   point for Rexarm
                if(homo_coor[1][0] < 0.02):
                    self.final_wp = self.rexarm.rexarm_IK(
                        [homo_coor[0][0] + 0.005, homo_coor[1][0] + 0.008, homo_coor[2][0], -90])
                else:
                    self.final_wp = self.rexarm.rexarm_IK(
                        [homo_coor[0][0] + 0.005, homo_coor[1][0] + 0.012, homo_coor[2][0], -90])
            else:
                self.final_wp = self.rexarm.rexarm_IK(
                    [homo_coor[0][0] + 0.015, homo_coor[1][0] + 0.002, homo_coor[2][0], -90])

            try:
                self.tp.set_final_wp(self.final_wp)
                self.tp.go()
            except TypeError:
                # not finding final_wp returns None,
                # which raises TypeError when indexing
                pass
        else:
            print(">>>>>>>>>>>>>>>> no tags, cannot move arm")
        time.sleep(1.0)
        # Rexarm moving home
        self.rexarm.set_positions([0.0, 0.0, 0.0, 0.0, 0.15])

        self.set_current_state("idle")

    def moving_arm_with_closed_gripper(self):
        """ Implement this function"""
        # self.rexarm.send_commands()
        if self.tags:
            self.tp.set_initial_wp()

            tagPose = self.tags[0].pose_t

            #################### FOR DEBUG ####################
            print(">>>>>>>>>>>>>>>> TAG POSE:")
            print(self.extrinsic_inv @
                  np.vstack((tagPose, np.array([1]))))
            ###################################################

            z = self.tags[0].pose_R[:, -1].reshape(3, 1)
            y = self.tags[0].pose_R[:, 1].reshape(3, 1)
            if y[0, 0] < 0:
                y *= -1
            tagPose += (CUBE_LEN) * z
            # tagPose -= 0.01 * y

            homo_coor = self.extrinsic_inv @\
                np.vstack((tagPose, np.array([1])))
            homo_coor = homo_coor / homo_coor[-1]
            homo_coor[2][0] += 0.035
            #################### FOR DEBUG ####################
            print(">>>>>>>>>>>>>>>> HOMO COOR:")
            print(homo_coor)
            ###################################################
            if (homo_coor[1][0] > -0.01):
                # add offsets to the tag to solve for a perfect desired
                #   point for Rexarm
                if(homo_coor[1][0] < 0.02):
                    self.final_wp = self.rexarm.rexarm_IK(
                        [homo_coor[0][0] + 0.035, homo_coor[1][0] + 0.008, homo_coor[2][0], -90])
                else:
                    self.final_wp = self.rexarm.rexarm_IK(
                        [homo_coor[0][0] + 0.035, homo_coor[1][0] + 0.012, homo_coor[2][0], -90])
            else:
                self.final_wp = self.rexarm.rexarm_IK(
                    [homo_coor[0][0] + 0.015, homo_coor[1][0] + 0.002, homo_coor[2][0], -90])

            try:
                self.tp.set_final_wp(self.final_wp)
                self.tp.go_with_closed_gripper()
            except TypeError:
                # not finding final_wp returns None,
                # which raises TypeError when indexing
                pass

        self.set_current_state("idle")

    def drop_block(self):
        self.rexarm.set_positions([0.0, 0.0, 0.0, -1.45, 0.15])
        time.sleep(1.0)
        self.rexarm.set_positions([0.0, 0.0, 0.0, -1.45, -1.5])
        time.sleep(1.0)

    def moving_mbot(self):
        """TODO: Implement this function"""
        self.publish_mbot_command(mbot_command_t.STATE_MOVING, (1, 1, 0), [
                                  (2, 2, 0), (1, 3, 0)])

    def moving_back(self):
        msg = robot_path_t()
        msg.utime = int(100000000 * time.time())
        msg.path_length = 2

        p1 = pose_xyt_t()
        p2 = pose_xyt_t()
        p1.x = 0.0
        p1.y = 0.0
        p1.theta = 0.0
        p2.x = 0.15
        p2.y = 0.0
        p2.theta = 0.0
        msg.path = [p1, p2]
        self.lc.publish("CONTROLLER_PATH", msg.encode())
        self.set_current_state("idle")

    def moving_to_block_side(self):
        msg = robot_path_t()
        msg.utime = int(1000000000 * time.time())
        msg.path_length = 4

        p1 = pose_xyt_t()
        p2 = pose_xyt_t()
        p3 = pose_xyt_t()
        p4 = pose_xyt_t()

        p1.x = 0.0
        p1.y = 0.0
        p1.theta = 0.0

        p2.x = 0.0
        p2.y = -0.4
        p2.theta = 0.0

        p3.x = 0.3
        p3.y = -0.4
        p3.theta = 0.0

        p4.x = 0.3
        p4.y = -0.265
        p4.theta = 0.0

        msg.path = [p1, p2, p3, p4]
        self.lc.publish("CONTROLLER_PATH", msg.encode())
        self.set_current_state("idle")

    def put_back_block(self):
        self.rexarm.set_positions(
            list(self.final_wp) + [self.rexarm.close_gripper()])
        time.sleep(1.0)
        self.rexarm.set_positions(
            list(self.final_wp) + [self.rexarm.open_gripper()])
        time.sleep(1.0)
        self.rexarm.set_positions(
            [1.5, 0.0, 0.0, 0.0, self.rexarm.open_gripper()])
        time.sleep(1.0)
        self.set_current_state("idle")

    def slampose_feedback_handler(self, channel, data):
        """
        Feedback Handler for slam pose
        this is run when a feedback message is recieved
        """
        msg = pose_xyt_t.decode(data)
        self.slam_pose = (msg.x, msg.y, msg.theta)

    def mbotstatus_feedback_handler(self, channel, data):
        """
        Feedback Handler for mbot status
        this is run when a feedback message is recieved
        """
        msg = mbot_status_t.decode(data)

    def get_mbot_feedback(self):
        """
        LCM Handler function
        Must be called continuously in the loop to get feedback.
        """
        self.lc.handle_timeout(10)

    def publish_mbot_command(self, state, goal_pose, obstacles):
        """
        Publishes mbot command.
        """
        msg = mbot_command_t()
        msg.utime = int(time.time() * 1e6)
        msg.state = state

        if state == mbot_command_t.STATE_STOPPED:
            pass
        elif state == mbot_command_t.STATE_MOVING:
            msg.goal_pose.x, msg.goal_pose.y, msg.goal_pose.theta = 1, 1, 1
            msg.num_obstacles = len(obstacles)
            for i in range(msg.num_obstacles):
                obs_pose = pose_xyt_t()
                obs_pose.utime = int(time.time() * 1e6)
                obs_pose.x, obs_pose.y, obs_pose.theta = obstacles[i]
                msg.obstacle_poses.append(obs_pose)
        else:
            raise NameError('Unknown mbot commanded state')

        self.lc.publish("MBOT_COMMAND", msg.encode())

    def arm_return_home(self):
        self.rexarm.set_positions([0.0, 0.0, -1.22, -np.pi / 3, 0.15])
        self.set_current_state("idle")

    """TODO: Add more functions and states in the state machine as needed"""

    def get_extrinsicInv(self):
        """
        Xucheng:

        Now it returns the hard-code ExtrinsicInv matrix
        """
        ################################################################
        # NOTE: CALCULATIONS BY HAND
        # rotation matrix
        theta = 19.0 * D2R

        alpha = -19.5 * D2R
        beta = 2.9 * D2R
        gamma = 0.0 * D2R

        trans_z = 0.073
        trans_x = 0.028

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha), np.cos(alpha)]])
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])
        temp = Rx @ Ry @ Rz
        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[0, :] = temp[2, :]
        rotation_matrix[1, :] = (-1) * temp[0, :]
        rotation_matrix[2, :] = (-1) * temp[1, :]
        tvec = np.array([0.028, -0.002, 0.073])
        extrinsicInv = np.zeros((4, 4))

        extrinsicInv[0:3, 0:3] = rotation_matrix
        extrinsicInv[0:3, 3] = np.squeeze(tvec)
        extrinsicInv[3, 3] = 1
        ################################################################
        # NOTE: CALCULATIONS BY solvePnP()
        #   Hard-coded.
        #   DEPRECATED DUE TO INACCURACY
        # extrinsic = np.array([[-8.27275930e-02, -9.96571841e-01, -8.42851409e-04, 5.19003157e-03],
        #                       [-2.70938513e-01, 2.33050726e-02, -
        #                           9.62314500e-01, 6.67610085e-02],
        #                       [9.59035175e-01, -7.93816013e-02, -
        #                           2.71937665e-01, -7.15535861e-02],
        #                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        ################################################################

        return extrinsicInv

    def get_extrinsic(self):
        return np.linalg.inv(self.get_extrinsicInv())
