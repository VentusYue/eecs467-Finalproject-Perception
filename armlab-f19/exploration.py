
"""
Pseudocode for algorithm

NOTE:
Maybe start in the same position and hard code in the corner positions?
Maybe need to modify motion_controller or exploration.cpp to receive goals and generate paths from that
Slam pose is received in state_machine

Task 1: Wander around map
0. Start by near supply and trash zones (STATE INITIAL SPIN)
    a. Spin around slowly to log april tags of supply and trash zones
    b. Save locations of zones
1. Wander around the map (STATE EXPLORING)
    a. Move from corners 1->2->3->4
        i. while number of block tags spotted > 0 (STATE FOUND BLOCK)
            I. Log current position as a checkpoint to return to once block is dropped off
            II. Move toward nearest block
            III. Goto Task 2 (STATE RETRIEVE AND PLACE BLOCK)
            IV. Return to checkpoint
        ii. At corner, spin in 360 degrees to check for tags
            I. If tag found, goto 1.a.i.I

Task 2: Pick up blocks
0. Position bot to face correct side of block
1. Pick up block
    a. Change state to moving_arm
2. Move to supply or trash area
    b. Change state to moving_mbot
3. Place block
    c. Change state to moving_arm
4. Change state to moving_mbot
5. Return to Task 1

Needed Vars:
checkpoint pose for Task 1 step 1.a.i.I
supply/trash poses for the drop off areas
corner locations
seen/unseen tags
state_machine
STATE

Need functions
Task 1
Task 2
spin mbot in place to see tags
rotate around block to see proper face
create path to pose
return to checkpoint
update seen april tags
convert from camera coordinates to global frame coordinates
Handlers for current robot pose
Broadcaster for goal for robot path
lcm handler
"""

import lcm
import numpy as np
import sys
import time
from lcmtypes import pose_xyt_t, robot_path_t
from math import sqrt, cos, sin

from enum import Enum

# Class for STATE enumerations


class State(Enum):
    INITIAL = 0
    EXPLORING = 1
    FOUND_BLOCK = 2
    RETRIVE_BLOCK = 3
    DETERMINE_BLOCK_PLACEMENT = 4
    TRASH = 5
    SUPPLY = 6
    DROP_BLOCK = 7
    RETURN_TO_CHECKPOINT = 8
    SPIN = 9
    COMPLETE = 10


D2R = 3.141592 / 180.0
R2D = 180.0 / 3.141592
PI = 3.141592


def clamp_radians(angle):
    if angle < -PI:
        while angle < -PI:
            angle += 2.0 * PI
    elif angle > PI:
        while angle > PI:
            angle -= 2.0 * PI
    return angle


def angle_diff_radians(angle1, angle2):
    diff = angle2 - angle1
    return clamp_radians(diff)


class Exploration():

    def __init__(self, state_machine):
        self.start_pose = np.array([0, 0, 0])
        self.current_pose = np.array([0, 0, 0])
        self.checkpoint = None  # Checkpoint pose for when mbot moves from Task 1 to Task 2
        self.trash_pose = np.array([0, -0.4, np.pi])
        self.supply_pose = np.array([0, 0, np.pi])

        self.colors = ["red", "orange", "green", "blue", "purple", "yellow"]
        # Store as color : locations in global frame
        self.new_block_tags = {}
        self.completed_block_tags = set()
        self.current_block = None
        self.current_goalPose = None
        self.current_block_idx = None

        self.sm = state_machine
        self.STATE = State.INITIAL

        self.lcm = lcm

        # TODO Need to set these corner locations to explore
        # self.corners = [None, None, None, None]
        self.corners = [self.start_pose]
        self.current_corner_goal = 0  # index of the current corner goal

        self.set_goal_for_moving_to = False

    def update_current_pose(self):
        self.current_pose = self.sm.slam_pose

    def robot_reached_goal(self, goal_pose, threshold=0.05, angle_threshold=0.05):
        """Returns True if the robot as reached the goal pose"""
        self.update_current_pose()
        goal_pose = np.array(goal_pose)

        # print("DISTANCE TO GOAL: {}".format(np.linalg.norm(
        #     self.current_pose[0:2] - goal_pose[0:2])))

        if np.linalg.norm(self.current_pose[0:2] - goal_pose[0:2]) > threshold:
            return False

        if abs(angle_diff_radians(self.current_pose[2], goal_pose[2])) > angle_threshold:
            return False
        return True

    def spin_in_place(self):
        """ Spin mbot in place """
        self.update_current_pose()
        # Send current pose coordinates + [pi/2, ]
        goals = np.arange(0, 420, 30, dtype=np.float) * \
            D2R + self.current_pose[2]
        threshold = 0.30
        for goal in goals:

            # Broadcast current goal pose
            goal_pose = []
            goal_pose.append(self.current_pose[0])
            goal_pose.append(self.current_pose[1])
            goal_pose.append(clamp_radians(goal))
            print("broadcasting pose:", goal_pose)
            self.broadcast_spin_goal(goal_pose)

            # While the robot has not reached the current goal pose, sleep
            while not self.robot_reached_goal(goal_pose):
                time.sleep(0.1)
                if self.current_goalPose is not None:
                    dist = np.linalg.norm(
                        self.current_pose[0:2] - self.current_goalPose[0:2])
                    # stop spinning
                    self.broadcast_spin_goal(
                        self.current_pose, only_one_pose=True)
                    if dist > threshold:
                        print(">>>>>>>>>>>>>> START MOVING TO BLOCK")
                        self.STATE = State.FOUND_BLOCK
                        print(">>>>>>>>>>>>>> ROBOT HAS FOUND A POSE!!")
                        # time.sleep(1.0)
                        return
                    print(">>>>>>>>>>>>> START PICKING UP THE BLOCK")
                    self.STATE = State.RETRIVE_BLOCK
                    return

            time.sleep(1.0)
        # When done spinning, set STATE to explore
        # self.STATE = State.EXPLORING

        # no tags found
        print(">>>>>>>>>>>>>> No tags found")
        self.STATE = State.COMPLETE

    def explore_corners(self):
        """
        Step 1: Have the robot explore the map by moving from corner to corner
        by broadcasting the next corner goal

        TODO Maybe add more waypoints?
        """

        # Step 1: Go from corner to corner of the map
        if self.current_corner_goal == len(self.corners):
            self.STATE = State.COMPLETE
            return
        corner = self.corners[self.current_corner_goal]

        # Broadcast current goal pose
        self.broadcast_goal(corner)

        # If the robot has reached the next corner, increment current_corner_goal
        if self.robot_reached_goal(corner):
            self.current_corner_goal += 1

            # Set state to SPIN to spin robot in place and take a look around
            self.STATE = State.SPIN

    def update_april_tags(self, tags, block_images):
        """
        Updates the currently seen block tags

        Will be called from control_station after detect_apriltags is called
        Set locations of blocks and supply areas
        """
        # tags-not-empty check
        if not tags:
            return

        threshold = 0.30

        for tag in tags:
            # IGNORE BIN AND SUPPLY AREA FOR DEMO!!!
            if tag.tag_id == 7:
                continue

            # Determine color of block associated with tag
            color = self.determine_color(tag, block_images)
            block_pose = self.convert_cam_coord_to_global(tag.pose_t)
            dist = np.linalg.norm(self.current_pose[0:2] - block_pose[0:2])

            if dist > threshold:  # or color == "NO COLOR":
                # self.new_block_tags[color] = block_pose
                if self.current_goalPose is None:
                    self.current_goalPose = block_pose
                    self.sm.current_goalPose = block_pose.reshape(3, 1)
                    self.sm.current_goalPose[2][0] = 0.254 / 2
                    print("Set block current_goalPose:", block_pose)
                    print("current color: ", color)
            else:
                if color == "supply" and self.supply_pose == None:
                    self.supply_pose = block_pose
                    print("Set supply pose:", self.supply_pose)
                    return
                if color == "trash" and self.trash_pose == None:
                    self.trash_pose = block_pose
                    print("Set trash pose:", self.trash_pose)
                    return

                # Update pose of block if block color has not already been completed
                # or it's pose does not exist in new_block_tags
                if color not in self.completed_block_tags and self.current_goalPose is None:
                    self.current_goalPose = block_pose
                    self.sm.current_goalPose = block_pose.reshape(3, 1)
                    self.sm.current_goalPose[2][0] = 0.254 / 2
                    print("Set block pose for color", color, ":", block_pose)

    def determine_color(self, tag, block_images):
        """
        Returns the color of the block associated with the tag

        May also return "supply" or "trash" to designate those respective areas
         - tag id 7 corresponds to supply and trash
         - tag ids 2, 3, 4, 5 are individual blocks
         - tag id 6 is long block

        TODO May need to make it so the robot moves around block to see colored face if that becomes an issue
        """

        # Supply and Trash
        if tag.tag_id == 7:
            if self.is_color_block_in_image(tag, "red", block_images[0]):
                return "supply"
            elif self.is_color_block_in_image(tag, "blue", block_images[4]):
                return "trash"
            else:
                print("APRIL TAG FOR SUPPLY/TRASH COULD NOT FIND VALID COLOR (RED/BLUE)")
                return "NO COLOR"

        # Other blocks
        for color in range(len(self.colors)):
            if self.is_color_block_in_image(tag, self.colors[color], block_images[color]):
                return self.colors[color]

        ############# FOR DEBUG #############
        # print("NO VALID COLOR FOUND FOR APRIL TAG")
        #####################################
        return "NO COLOR"

    # Search within 50? pixels of the center for color

    def is_color_block_in_image(self, tag, color, block_image):
        limit = 50
        xMin = max(0, int(tag.center[0] - limit))
        xMax = min(639, int(tag.center[0] + limit))
        yMin = max(0, int(tag.center[1] - 50))
        yMax = min(479, int(tag.center[1] + 50))
        # print(tag.center)
        # print(xMin,xMax,yMin,yMax)
        # if color == "purple":
        # print(block_image[xMin:xMax,yMin:yMax,:])
        # NOTE: not sure what it does
        #   but rewriting it in the following way should be faster
        ################################################################
        # for x in range(xMin, xMax):
        #     for y in range(yMin, yMax):
        #         if np.sum(block_image[x][y]) > 0:
        #             return True
        ################################################################
        # if np.sum(block_image[xMin:xMax, yMin:yMax, :]) > 0:
        #     return True
        if np.sum(block_image[yMin:yMax, xMin:xMax, :]) > 0:
            return True
        ################################################################

        return False

    def move_to_block(self):
        """
        If there are valid block april tags, select block and move towards it
        TODO Change when this is called so it can interrupt exploration and spinning
        """

        if self.current_goalPose is not None:
            # Broadcast block goal
            if not self.set_goal_for_moving_to:
                self.set_goal_for_moving_to = True
                # self.current_block = self.find_min_block()
                # print("current block indx: ", self.current_block)
                # try:
                #     self.current_goalPose = \
                #         self.new_block_tags.pop(self.current_block)
                # except KeyError:
                #     print(">>>>>>>>>>>>>> MOVE TO BLOCK: key not found for pop()")
                print("move to block publishing goal pose ",
                      self.get_astar_goal())
                self.broadcast_goal(self.get_astar_goal())

            # Check if we have reached the block
            if self.robot_reached_goal(self.get_astar_goal()):
                # SET STATE TO RETRIEVE BLOCK
                self.broadcast_spin_goal(
                    self.current_pose, only_one_pose=True)
                self.current_goalPose = None
                self.set_goal_for_moving_to = False
                time.sleep(2.0)
                if self.current_goalPose is not None:
                    self.STATE = State.RETRIVE_BLOCK
                    self.current_goalPose = None
                else:
                    self.STATE = State.SPIN
                # Remove block from new_block_tags and add to completed_block_tags
                # self.completed_block_tags.add(self.current_block)

    def retrieve_block(self):
        """
        Pick up the current block

        When block has been retrieved, set state to DETERMINE BLOCK PLACEMENT

        UPDATE checkpoint

        Deal with different types of blocks (single, triple, wedged in corner, along wall)
        """
        self.sm.set_current_state("moving_arm")
        time.sleep(5.0)
        self.STATE = State.DROP_BLOCK
        print("start dropping block")

    def drop_block(self):
        """
        Drop off block held

        Upon completion, set state to RETURN TO CHECKPOINT
        """
        self.broadcast_goal(self.trash_pose)
        if (self.robot_reached_goal(self.trash_pose)):
            self.sm.set_current_state("drop_block")
            self.STATE = State.COMPLETE

    def determine_block_placement(self):
        """
        TODO If the block belongs in trash, set state to MOVING TO TRASH

        TODO If the block belongs in supply, set state to MOVING TO SUPPLY
        """
        pass

    # DONE
    def move_block_to_trash_supply(self):

        # Set goal to TRASH or SUPPLY depending on current state
        if self.STATE == State.TRASH:
            goal_pose = self.trash_pose
        elif self.STATE == State.SUPPLY:
            goal_pose = self.supply_pose
        else:
            # SHOULD NEVER GET HERE
            print("ERROR! INVALID STATE!\n EXPECTED TRASH OR SUPPLY")
            exit(1)

        # Broadcast goal
        self.broadcast_goal(goal_pose)
        if self.robot_reached_goal(goal_pose):

            # SET STATE TO DROP BLOCK
            self.STATE = State.DROP_BLOCK

    def return_to_checkpoint(self):
        """ Return robot to checkpoint """

        if self.checkpoint == None:
            print("Uninitialized checkpoint! Exiting...")
            exit(1)

        self.broadcast_goal(self.checkpoint)
        if self.robot_reached_goal(self.checkpoint):

            # SET STATE TO EXPLORING
            self.STATE = State.EXPLORING
            pass

    def convert_cam_coord_to_global(self, cam_pose):
        """Converts cam_pose to global coordinate frame"""

        # Convert from cam to rexarm
        extrinsic_inv = self.sm.get_extrinsicInv()

        pose_homo = np.append(cam_pose, [1])
        arm_pose = np.dot(extrinsic_inv, pose_homo)

        # Convert from rexarm to mbot frame
        # Shift +11.2 cm in x
        mbot_pose = arm_pose
        mbot_pose[0] += 0.112

        # Convert from mbot to global
        try:
            theta = self.sm.slam_pose[2]
            rot_mat = np.array(
                [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]], dtype=np.float)
            global_pose = np.array(
                [mbot_pose[0], mbot_pose[1]], dtype=np.float)
            global_pose = rot_mat @ global_pose
            global_pose[0] += self.sm.slam_pose[0]
            global_pose[1] += self.sm.slam_pose[1]
            global_pose = list(global_pose)
            global_pose.append(theta)
            return np.array(global_pose)
        except TypeError:
            # currently slam pose available
            return None

    def broadcast_spin_goal(self, goal_pose, only_one_pose=False):
        """ Broadcase robot_path_t on channel CONTROLLER_PATH"""
        msg = robot_path_t()
        msg.utime = int(time.time() * 1000000)

        start_pose = pose_xyt_t()
        start_pose.x = self.current_pose[0]
        start_pose.y = self.current_pose[1]
        start_pose.theta = self.current_pose[2]
        end_pose = pose_xyt_t()
        end_pose.x = goal_pose[0]
        end_pose.y = goal_pose[1]
        end_pose.theta = goal_pose[2]
        if not only_one_pose:
            msg.path = [start_pose, end_pose]
        else:
            msg.path = [start_pose]
        msg.path_length = len(msg.path)
        self.sm.lc.publish("CONTROLLER_PATH", msg.encode())

    def broadcast_goal(self, goal_pose):
        """Broadcast goal_pose on channel GOAL_POSE"""
        msg = pose_xyt_t()
        msg.utime = int(time.time() * 1000000)
        msg.x = goal_pose[0]
        msg.y = goal_pose[1]
        msg.theta = goal_pose[2]
        self.sm.lc.publish("GOAL_POSE", msg.encode())

    def find_min_block(self):
        """Finds the index of the closest block that hasn't been picked up"""
        self.update_current_pose()
        min_dist = sys.float_info.max
        indx = 0
        for key, val in self.new_block_tags.items():
            if val is None:
                continue
            dx = self.current_pose[0] - val[0]
            dy = self.current_pose[1] - val[1]
            dist = sqrt(dx ** 2 + dy ** 2)
            if dist < min_dist:
                indx = key
                min_dist = dist

        return indx

    def get_astar_goal(self):
        dist = \
            np.linalg.norm(self.current_pose[0:2] - self.current_goalPose[0:2])

        fraction = 1 - (0.11 + 0.12) / dist

        return self.current_pose + (self.current_goalPose - self.current_pose) * fraction
