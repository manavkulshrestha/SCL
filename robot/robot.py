import os

from absl import app
from absl import flags

import numpy as np

from robot.gripper import Suction
import pybullet as p
import pybullet_data

import time

UR5_URDF_PATH = 'ur5/ur5.urdf'
ASSET_ROOT = 'robot/assets/'

HOME_J = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi


class UR5:
    def __init__(self, base_pos):
        self.id = p.loadURDF(os.path.join(ASSET_ROOT, UR5_URDF_PATH), base_pos)

        ddict = {'fixed': [], 'rigid': [], 'deformable': []}
        self.ee_id = 10
        self.ee = Suction(ASSET_ROOT, self.id, self.ee_id-1, ddict)

        self.n_joints = p.getNumJoints(self.id)
        joints = [p.getJointInfo(self.id, i) for i in range(self.n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        for i in range(len(self.joints)):
            p.resetJointState(self.id, self.joints[i], HOME_J[i])

        self.ee.release()

        # self.ee_position =

    def accurateCalculateInverseKinematics(self, targetPos, threshold, maxIter=100):
        closeEnough = False
        iter2 = 0

        dist2 = 1e30
        while not closeEnough and iter2 < maxIter:
            jointPoses = p.calculateInverseKinematics(self.id, self.ee_id, targetPos)
            self.set_joints(jointPoses)

            ls = p.getLinkState(self.id, self.ee_id)
            newPos = ls[4]
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])

            closeEnough = (dist2 < threshold)
            iter2 = iter2 + 1

        return jointPoses

    def move_ee(self, pos, orn=None, error_thresh=0.0005):
        q_tar = np.array(p.calculateInverseKinematics(self.id, self.ee_id, pos, targetOrientation=orn))

        # while True:
        #     q_cur = np.array([q_state[0] for q_state in p.getJointStates(self.id, self.joints)])
        #     err = np.linalg.norm(q_tar - q_cur)
        #
        #     print(f'{q_tar=}')
        #     print(f'{q_cur=}')
        #     print(f'{err=}')
        #
        #     if err < error_thresh:
        #         break
        #
        #     p.setJointMotorControlArray(bodyUniqueId=self.id,
        #                                 jointIndices=self.joints,
        #                                 controlMode=p.POSITION_CONTROL,
        #                                 targetPositions=q_tar)
        #
        #     p.stepSimulation()
        #     time.sleep(1 / 24)
        #
        # return q_tar, q_cur

        self.set_joints(q_tar)

        return q_tar, q_tar

    def suction(self, on):
        if on:
            self.ee.activate()
        else:
            self.ee.release()

    def set_joints(self, q):
        for ji, qi in zip(self.joints, q):
            p.resetJointState(self.id, ji, qi)

    @property
    def ee_pose(self):
        return p.getLinkState(self.id, self.ee_id)[:2]

    @property
    def ee_offset(self):
        return p.getLinkState(self.id, self.ee_id)[2:4]

    @property
    def ee_frame(self):
        return p.getLinkState(self.id, self.ee_id)[4:6]

    # (-0.017928933565927566, 0.1091351801612924, 0.30883961078140254), (-3.6392306798307536e-05, 0.01339669391733023, 1.7864816034829367e-05, 0.9999102594475835)
    # (0.0, 0.0, 0.0), (0.7071067811882787, -0.7071067811848163, 7.31230107716731e-14, -7.312301077203115e-14)
    # (-0.01792893372476101, 0.10913518071174622, 0.3088396191596985), (0.707055926322937, -0.707030713558197, -0.009447161108255386, 0.009498627856373787)


def setup():
    ur5 = UR5([-0.5, 0, 0])

    for _ in range(100):
        p.stepSimulation()

    time.sleep(10)


def main():
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    target = (-0.07796166092157364, 0.005451506469398737, -0.06238798052072525)
    dist = 1.0
    yaw = 89.6000747680664
    pitch = -17.800016403198242
    p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")

    setup()


if __name__ == '__main__':
    main()
