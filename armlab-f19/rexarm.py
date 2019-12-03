# rexarm.py
import numpy as np
import time
import math

# NOTE - Zhihao Ruan
#   Please import cos() and sin() from numpy
#   for better matrix/array manipulation
from numpy import cos
from numpy import sin

"""

Implement the missing functions
add anything you see fit

"""

""" Radians to/from  Degrees conversions """
D2R = 3.141592 / 180.0
R2D = 180.0 / 3.141592


def clamp_radians(theta):
    return np.clip(theta, -np.pi, np.pi)


def get_cos(a, b, c):
    return math.acos((a**2 + b**2 - c**2) / (2 * a * b))


class Rexarm():
    def __init__(self, joints):
        self.joints = joints
        self.gripper = joints[-1]
        self.gripper_open_pos = np.deg2rad(-60.0)
        self.gripper_closed_pos = np.deg2rad(0.0)
        self.gripper_state = True
        self.estop = False
        # Find the physical angle limits of the Rexarm. Remember to keep track of this if you include more motors
        self.angle_limits = np.array([
            [-104.84, -100.44, -128.30, -88.00, -90.00],
            [106.89, 96.04, 119.21, 103.67, 10]], dtype=np.float) * D2R

        """ Commanded Values """
        self.num_joints = len(joints)
        self.position = [0.0] * self.num_joints     # degrees
        self.speed = [1.0] * self.num_joints        # 0 to 1
        self.max_torque = [1.0] * self.num_joints   # 0 to 1

        """ Feedback Values """
        self.joint_angles_fb = [0.0] * self.num_joints  # degrees
        self.speed_fb = [0.0] * self.num_joints        # 0 to 1
        self.load_fb = [0.0] * self.num_joints         # -1 to 1
        self.temp_fb = [0.0] * self.num_joints         # Celsius
        self.move_fb = [0] * self.num_joints

        """ Arm Lengths """

        # Fill in the measured dimensions.
        # self.base_len = 0.023  # NOTE: center of 1st motor - 2nd motor; deprecated
        self.base_len = 0.11  # center of 2nd motor to ground
        self.shoulder_len = 0.055
        self.elbow_len = 0.056
        self.wrist_len = 0.126

        """ DH Table """
        # Fill in the variables.
        # NOTE: The z axis should always be the axis of rotation of the motor!
        self.dh_table = [{"d": self.base_len, "a": 0, "alpha": 90 * D2R},
                         {"d": 0, "a": self.shoulder_len, "alpha": 0},
                         {"d": 0, "a": self.elbow_len, "alpha": 0},
                         {"d": 0, "a": self.wrist_len, "alpha": 0}]

    def initialize(self):
        pos = np.array([0.0, 0.0, -70.0, -60.0, 8.0], dtype=np.float) * D2R
        for i, joint in enumerate(self.joints):
            joint.enable_torque()
            try:
                joint.set_position(pos[i])
            except:
                print("initialize joint", i, "falied")
                exit(1)
            
            joint.set_torque_limit(0.5)
            joint.set_speed(0.25)

    def open_gripper(self):
        """
        Xucheng:
        Return the joint angle needed to open gripper
        """
        if self.gripper_state:
            self.gripper_state = False
        return -1.5

    def close_gripper(self):
        """
        Xucheng:
        Return the joint angle needed to close gripper 
        """
        if not self.gripper_state:
            self.gripper_state = True
        return 0.20

    def set_positions(self, joint_angles, update_now=True):
        joint_angles = self.clamp(joint_angles)
        # print("joint angles: ", joint_angles)
        for i, joint in enumerate(self.joints):
            self.position[i] = joint_angles[i]
            if(update_now):
                try:
                    joint.set_position(joint_angles[i])
                except:
                    print("joint idx: ", i, "set position failed")

    def set_speeds_normalized_global(self, speed, update_now=True):
        for i, joint in enumerate(self.joints):
            self.speed[i] = speed
            if(update_now):
                joint.set_speed(speed)

    def set_speeds_normalized(self, speeds, update_now=True):
        for i, joint in enumerate(self.joints):
            self.speed[i] = speeds[i]
            if(update_now):
                joint.set_speed(speeds[i])

    def set_speeds(self, speeds, update_now=True):
        for i, joint in enumerate(self.joints):
            self.speed[i] = speeds[i]
            speed_msg = abs(speeds[i] / joint.max_speed)
            if (speed_msg < 3.0 / 1023.0):
                speed_msg = 3.0 / 1023.0
            if(update_now):
                joint.set_speed(speed_msg)

    def set_torque_limits(self, torques, update_now=True):
        for i, joint in enumerate(self.joints):
            self.max_torque[i] = torques[i]
            if(update_now):
                joint.set_torque_limit(torques[i])

    def send_commands(self):
        self.set_positions(self.position)
        self.set_speeds_normalized(self.speed)
        self.set_torque_limits(self.max_torque)

    def enable_torque(self):
        for joint in self.joints:
            joint.enable_torque()

    def disable_torque(self):
        for joint in self.joints:
            joint.disable_torque()

    def get_positions(self):
        for i, joint in enumerate(self.joints):
            try:
                self.joint_angles_fb[i] = joint.get_position()
            except:
                print("joint idx: ", i, "get position failed")
        return self.joint_angles_fb

    def get_speeds(self):
        for i, joint in enumerate(self.joints):
            try:
                self.speed_fb[i] = joint.get_speed()
            except:
                print("joint idx: ", i, "get speed failed")
        return self.speed_fb

    def get_loads(self):
        for i, joint in enumerate(self.joints):
            try:
                self.load_fb[i] = joint.get_load()
            except:
                print("joint idx: ", i, "get load failed")
        return self.load_fb

    def get_temps(self):
        for i, joint in enumerate(self.joints):
            try:
                self.temp_fb[i] = joint.get_temp()
            except:
                print("joint idx: ", i, "get temp failed")
        return self.temp_fb

    def get_moving_status(self):
        for i, joint in enumerate(self.joints):
            try:
                self.move_fb[i] = joint.is_moving()
            except:
                print("joint idx: ", i, "get is_moving failed")
        return self.move_fb

    def get_feedback(self):
        self.get_positions()
        self.get_speeds()
        self.get_loads()
        self.get_temps()
        self.get_moving_status()

    def pause(self, secs):
        time_start = time.time()
        while((time.time() - time_start) < secs):
            self.get_feedback()
            time.sleep(0.05)
            if(self.estop == True):
                break

    def clamp(self, joint_angles):
        """
        DONE: Implement this function to clamp the joint angles
        """
        # print(joint_angles)
        for i in range(len(joint_angles)):
            if joint_angles[i] > self.angle_limits[1][i]:
                joint_angles[i] = self.angle_limits[1][i]
            elif joint_angles[i] < self.angle_limits[0][i]:
                joint_angles[i] = self.angle_limits[0][i]
        return joint_angles

    def calc_A_FK(self, theta, link):
        """
        Implement this function
        theta is radians of the link number
        link is the index of the joint, and it is 0 indexed (0 is base, 1 is shoulder ...)
        returns a matrix A(2D array)
        """
        # A = ROT_z,theta * Trans_z,d * Trans_x,a * ROT_x,alpha
        alpha = self.dh_table[link]["alpha"]
        a = self.dh_table[link]["a"]
        d = self.dh_table[link]["d"]

        rot_ztheta = np.array([[cos(theta), -sin(theta), 0, 0],
                               [sin(theta), cos(theta), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float)

        trans_zd = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, d],
                             [0, 0, 0, 1]], dtype=np.float)

        trans_xa = np.array([[1, 0, 0, a],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.float)

        rot_xalpha = np.array([[cos(alpha), 0, -sin(alpha), 0],
                               [0, 1, 0, 0],
                               [sin(alpha), 0, cos(alpha), 0],
                               [0, 0, 0, 1]], dtype=np.float)

        A = rot_ztheta @ trans_zd @ trans_xa @ rot_xalpha

        ########## FOR DEBUG ##########
        # print(A)
        # print(link, "theta: ", theta)
        # print(rot_ztheta @ trans_zd)
        ###############################

        return A

    def rexarm_FK(self, joint_num=4):
        """
        Implement this function

        Calculates forward kinematics for rexarm
        takes a DH table filled with DH parameters of the arm
        and the link to return the position for
        returns a 4-tuple (x, y, z, phi) representing the pose of the
        desired link
        """

        # endpoint of the grabber with respect to its own coordinate frame
        endpoint = np.array([[0, 0, 0, 1]], dtype=np.float).T
        angles = self.joint_angles_fb
        for i in range(joint_num - 1, -1, -1):
            endpoint = self.calc_A_FK(angles[i], i) @ endpoint

        endpoint = endpoint / endpoint[-1]

        # Rotate around z axis by 90 degrees
        # NOTE: By default the y axis is encoded as the forward direction,
        #   which is not in accoradance with our coordinate frame. We need
        #   to fix that.
        theta = 90 * D2R
        endpoint = np.array([[cos(theta), -sin(theta), 0, 0],
                             [sin(theta), cos(theta), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.float) @ endpoint

        return tuple(endpoint.reshape(1, -1)[0])

    def rexarm_IK(self, pose):
        """
        Implement this function
        Calculates inverse kinematics for the rexarm
        pose is a tuple (x, y, z, phi) which describes the desired
        end effector position and orientation.
        If the gripper is perpendicular to the floor and facing down,
        then phi is -90 degree.
        If the gripper is parallel to the floor,
        then phi is 0 degree.
        returns a 4-tuple of joint angles or None if configuration is impossible
        """

        """
        Notes from Xucheng:
        I assume phi is either -90 degree (normal reach) or 0 degree (reaching high)
        """
        # print("start IK solver, solve for pose: ", pose)
        x, y, z, phi = pose[0], pose[1], pose[2], pose[3]
        R = math.sqrt(x ** 2 + y ** 2)
        theta_1 = math.atan2(y, x)
        if(phi != -90 and phi != 0):
            print("Return None. Phi has to be either -90 or 0!")
            return None
        elif(phi == -90):  # normal reach
            if z > self.base_len + self.shoulder_len + self.elbow_len - self.wrist_len:
                print("IK solver finds no solution due to invalid z!")
                return None
            M = math.sqrt(R**2 + (self.base_len - z)**2)
            if M < abs(self.shoulder_len - self.elbow_len):
                # very unlikely to happen
                print("IK solver finds no solution! Too close!")
                return None
            if M > self.shoulder_len + self.elbow_len:
                # switch to reaching far
                print("Switch to reaching far")
                if(M > self.shoulder_len + self.elbow_len + self.wrist_len):
                    print("IK solver find no solution! Too far!")
                    return None
                print("IK solver is trying to find solution for reaching far")
                alpha = math.atan2(R, self.base_len - z)
                beta = get_cos(M, self.shoulder_len +
                               self.elbow_len, self.wrist_len)
                gamma = get_cos(self.shoulder_len +
                                self.elbow_len, self.wrist_len, M)
                theta_2 = -(math.pi - alpha - beta)
                theta_3 = 0.0
                theta_4 = -(math.pi - gamma)
            else:
                print("IK solver is trying to find solution for normal reach")
                alpha = math.atan2(self.wrist_len + z - self.base_len, R)
                M = math.sqrt(R**2 + (self.wrist_len + z - self.base_len)**2)
                gamma = get_cos(self.shoulder_len, self.elbow_len, M)
                beta = get_cos(self.shoulder_len, M, self.elbow_len)
                theta_2 = -(math.pi / 2 - alpha - beta)
                theta_3 = -(math.pi - gamma)
                theta_4 = -(alpha + beta + gamma - math.pi / 2)
        elif(phi == 0):  # reaching high
            if z < self.base_len or z > self.base_len + self.shoulder_len + self.elbow_len:
                print("IK solver finds no solution due to invalid z!")
                return None
            M = math.sqrt((z - self.base_len)**2 + (R - self.wrist_len)**2)
            if R < self.wrist_len - self.shoulder_len:
                print("IK solver finds no solution! Too close to Mbot!")
                return None
            if M > self.shoulder_len + self.elbow_len or R > self.elbow_len + self.wrist_len:
                print("IK solver finds no solution! Too far!")
                return None
            if M < abs(self.shoulder_len - self.elbow_len):
                print("IK solver finds no solution!")
            print("IK solver is trying to find solution for reaching high")
            beta_2 = math.atan2(self.wrist_len - R, z - self.base_len)
            alpha_2 = math.atan2(z - self.base_len, self.wrist_len - R)
            alpha_1 = get_cos(self.elbow_len, M, self.shoulder_len)
            beta_1 = get_cos(self.shoulder_len, M, self.elbow_len)
            gamma = get_cos(self.shoulder_len, self.elbow_len, M)
            theta_2 = beta_1 + beta_2
            theta_3 = -(math.pi - gamma)
            theta_4 = -(math.pi - alpha_1 - alpha_2)
        angles = [theta_1, theta_2, theta_3, theta_4]
        # print("testing if ", angles, "lies in joint limits")
        in_limits = True
        for i in range(4):
            if(angles[i] < self.angle_limits[0][i] or angles[i] > self.angle_limits[1][i]):
                # print("Joint angle", i, "invalid: ",
                #       angles[i], " limits: ", self.angle_limits[0][i], self.angle_limits[1][i])
                in_limits = False
                break
        if in_limits:
            return tuple([theta_1, theta_2, theta_3, theta_4])
        else:
            print("IK solver finds no solution!")
            return None


"""
def main():
    arm  = Rexarm([0, 1, 2, 3])

    angles = [0.03, 0.03, -0.05, -90]
    print("test1: x y z phi", angles)
    ik_result = arm.rexarm_IK(angles)
    if ik_result == None:
        print("None")
    else:
        fk_result = arm.rexarm_FK(ik_result)
    print(fk_result, "\n")

    angles = [0.08, 0.08, -0.07, -90]
    print("test1: x y z phi", angles)
    ik_result = arm.rexarm_IK(angles)
    if ik_result == None:
        print("None")
    else:
        fk_result = arm.rexarm_FK(ik_result)
    print(fk_result, "\n")

    angles = [0.055, 0.055, 0.10, 0]
    print("test2: x y z phi", angles)
    ik_result = arm.rexarm_IK(angles)
    if ik_result == None:
        print("None")
    else:
        fk_result = arm.rexarm_FK(ik_result)
    print(fk_result, "\n")

if __name__ == "__main__":
    main()
"""
