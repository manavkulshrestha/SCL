import os
from typing import Iterable

from absl import app
from absl import flags

import numpy as np

from robot.gripper import Suction
import pybullet as p
import pybullet_data

import time

# from utility import check_convergence, unit_vec, draw_sphere_marker


# UR5_URDF_PATH = 'ur5/ur5.urdf'
# ASSET_ROOT = 'assets/'

UR5_URDF_PATH = 'ur5/ur5.urdf'
ASSET_ROOT = '../robot/assets/'


class UR5:
    def __init__(self, base_pos, move_timestep=0):
        pth = os.path.join(ASSET_ROOT, UR5_URDF_PATH)
        self.id = p.loadURDF(pth, base_pos)

        ddict = {'fixed': [], 'rigid': [], 'deformable': []}
        self.ee_id = 10
        self.ee = Suction(ASSET_ROOT, self.id, self.ee_id-1, ddict)

        self.n_joints = p.getNumJoints(self.id)
        joints = [p.getJointInfo(self.id, i) for i in range(self.n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        self.home_q = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.set_q(self.home_q)

        self.ee.release()
        self.move_timestep = move_timestep

    def set_q(self, q):
        for ji, qi in zip(self.joints, q):
            p.resetJointState(self.id, ji, qi)


    def ik(self, pos, orn):
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.id,
            endEffectorLinkIndex=self.ee_id,
            targetPosition=pos,
            targetOrientation=orn,
            lowerLimits=[-3*np.pi/2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi/2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.home_q).tolist(),
            maxNumIterations=200,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        # joints[2:] = (joints[2:]+np.pi) % (2*np.pi) - np.pi
        return joints

    @classmethod
    def _norm(cls, it: Iterable) -> float:
        return np.linalg.norm(it)

    @classmethod
    def _unit_vec(cls, lst: np.ndarray) -> np.ndarray:
        mag = cls._norm(lst)
        return (lst / mag) if mag > 0 else 0

    def move_q(self, tar_q, error_thresh=1e-2, speed=0.01, break_cond=lambda: False, max_iter=300, **kwargs):
        i = 0
        assert i < max_iter
        while i < max_iter:
            cur_q = np.array([p.getJointState(self.id, i)[0] for i in self.joints])
            err_q = tar_q - cur_q
            if break_cond() or (np.abs(err_q) < error_thresh).all():
                # p.removeBody(marker)
                return True, tar_q, cur_q

            u = self._unit_vec(err_q)
            step_q = cur_q + u * speed
            p.setJointMotorControlArray(
                bodyIndex=self.id,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=step_q,
                positionGains=np.ones(len(self.joints)))
            p.stepSimulation()
            i += 1
            time.sleep(self.move_timestep)

        # p.removeBody(marker)
        return False, tar_q, cur_q

    def move_ee(self, pos, orn=None, error_thresh=1e-2, speed=0.01, break_cond=lambda: False, max_iter=300, **kwargs):
        tar_q = self.ik(pos, orn)
        # marker = draw_sphere_marker(pos)
        self.move_q(tar_q, error_thresh=error_thresh, speed=speed, break_cond=break_cond, max_iter=max_iter, **kwargs)

    def move_ee_down(self, pos, orn=(0, 0, 0, 1), **kwargs):
        """
        moves down from `pos` to z=0 until it detects object
        returns: pose=(pos, orn) at which it detected contact"""
        pos = [*pos[:2], 0]
        self.move_ee(pos, orn=orn, break_cond=self.ee.detect_contact, **kwargs)
        return self.ee_pose

    def move_ee_above(self, pos, orn=(0, 0, 0, 1), above_offt=(0, 0, 0.2), **kwargs):
        a_pos = np.add(pos, above_offt)
        self.move_ee(a_pos, orn=orn, **kwargs)

    def move_ee_away(self, offt):
        ee_pos, ee_orn = self.ee_pose
        target_pos = np.add(ee_pos, offt)
        self.move_ee(target_pos, ee_orn)

    def suction(self, on):
        if on:
            self.ee.activate()
            p.setCollisionFilterGroupMask(self.ee.check_grasp(), -1, 0, 0)
        else:
            p.setCollisionFilterGroupMask(self.ee.check_grasp(), -1, 1, 1)
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


def setup(pos=[-0.5, 0, 0]):
    ur5 = UR5(pos)

    for _ in range(100):
        p.stepSimulation()

    return ur5


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

    robot = setup()
    # while True:
    #     print('> ', end='') # -1 -0.5 0.5 -0.5 -0.5 0
    #     q_nopi = [float(x) for x in input().split(' ')]
    #     # print(l)
    #     # robot.set_q(list(np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi))
    #     q = np.array(q_nopi)*np.pi
    #     robot.set_q(q)
    #     print(q)
    #     print(robot.ee_pose)

    file = np.load('../demo/test.npz')
    qs = file['qs']

    for q in qs:
        robot.set_q(q)
        time.sleep(0.1)


# [-1, -0.5, 0.5, -0.5, -0.5, 0]
if __name__ == '__main__':
    main()
