import time

import numpy as np
import rtde_control
import rtde_receive
import robotiq_gripper


UR5_IP = '192.168.0.33'
s = 0.5
a = 0.3

up_q = np.array([-1, -0.5, 0, -0.5, -0.5, 0]) * np.pi
home_q = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi


def main():
    print('Establishing interface')
    rtde_c = rtde_control.RTDEControlInterface(UR5_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(UR5_IP)

    print("Creating gripper...")
    gripper = robotiq_gripper.RobotiqGripper()
    print("Connecting to gripper...")
    gripper.connect(UR5_IP, 63352)
    # time.sleep(10)

    print("Activating gripper...")
    gripper.suction_setup()
    time.sleep(10)
    # gripper.suction(False)
    gripper.suction(True)
    time.sleep(10)
    gripper.suction(False)
    # gripp
    # time.sleep(5)
    # time.sleep(5)
    # gripper.activate(auto_calibrate=True)
    # time.sleep(5)
    # gripper

    # print('Getting current configuration')
    # print('Doing something')
    actual_q = rtde_r.getActualQ()
    print(actual_q)
    # rtde_c.moveJ(up_q, s, a)
    rtde_c.moveJ(home_q, s, a)
    #
    # print('done')


if __name__ == '__main__':
    main()
