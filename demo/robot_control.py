import time
from collections import OrderedDict
from socket import socket, AF_INET, SOCK_STREAM

import numpy as np
import rtde_control
import rtde_receive
import robotiq_gripper

from typing import List

from demo.VacuumGripper import VacuumGripper

UR5_IP = '192.168.0.33'
s = 0.5
a = 0.3

up_q = np.array([-1, -0.5, 0, -0.5, -0.5, 0]) * np.pi
home_q = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi


table_q = np.array([
    [[-4.679103676472799, -0.6932582420161744, 2.162341896687643, -3.057669778863424, -1.5295241514789026, 1.0650970935821533], [-2.186981980000631, -0.7352921527675171, 2.2481892744647425, -3.0904442272581996, -1.5628483931170862, 1.4174118041992188]],
    [[-3.845755402241842, -0.1504940551570435, 0.6257465521441858, -2.025536676446432, -1.5984151999102991, 2.4269793033599854], [-2.7514565626727503, -0.23492176950488286, 0.7924483458148401, -2.143445154229635, -1.5566309134112757, 3.553144931793213]]
])

table_x = np.array([
    [[0.15373595058918, 0.11967502534389496, -0.06333543360233307, 1.5273715257644653, 2.69427227973938, 0.009868443943560123], [0.36965832114219666, 0.9747239351272583, -0.037161193788051605, -2.0613481998443604, -2.3678438663482666, -0.015499585308134556]],
    [[1.0393606424331665, 0.041623637080192566, -0.06860598176717758, -2.1963465213775635, -2.2200493812561035, 0.04543614760041237], [1.0429596900939941, 0.9402209520339966, -0.032180801033973694, -2.2395496368408203, -2.1923282146453857, -0.03029365837574005]]
])


def suction_test(con, rec, gripper):
    teach_move(con, rec)
    gripper.suction(True)
    teach_move(con, rec)
    gripper.suction(False)


def teach_mode_wait(rtde_c, rtde_r, string):
    rtde_c.teachMode()
    input(string)
    rtde_c.endTeachMode()

    q = rtde_r.getActualQ()
    print(f'Q at end of teach mode: {q}')
    x = rtde_c.getForwardKinematics(q)
    print(f'X at end of teach mode: {x}')


def print_table():
    for row in table_x:
        row = np.array([r[:3] for r in row])
        p = lambda x: '('+(', '.join(str(xi) for xi in x))+')'
        print(f'{p(row[0].round(4))}, {p(row[1].round(4))}')


def test(con, rec):
    su = np.zeros(6)
    for row in table_x:
        for xij in row:
            su += xij

    print(su)
    su /= 4
    print(su)
    su[2] += 0.2

    print(su)  # [ 0.65142865  0.51906089  0.04967915 -1.24246821 -1.0214873   0.00237784]
    time.sleep(10)
    con.moveJ_IK(table_x[0, 0] + [0, 0, 0.2, 0, 0, 0])


def pos_readout(con, rec, dec=4):
    con.teachMode()

    while True:
        q = np.array(rec.getActualQ())
        x = np.array(con.getForwardKinematics(q))
        print(f'q: {q.round(dec)}, x: {x.round(4)}')
        time.sleep(0.1)

    con.endTeachMode()


def teach_move(con, rec, wait_after=5):
    con.teachMode()
    input('End?')
    con.endTeachMode()
    time.sleep(wait_after)


def teach_move_timed(con, rec, wait_after=5, timeout=5):
    con.teachMode()
    time.sleep(5)
    con.endTeachMode()
    time.sleep(wait_after)


def pos_record(path, con, rec, timeout=30):
    start = time.time()
    con.teachMode()

    qs, xs = [], []
    print('recording now...', end='')
    while (time.time() - start) < 30:
        q = np.array(rec.getActualQ())
        x = np.array(con.getForwardKinematics(q))

        qs.append(q)
        xs.append(x)

        time.sleep(0.1)

    print('done')
    con.endTeachMode()
    np.savez(path, qs=qs, xs=xs)


# teach_mode_wait(con, rec, 'End?')
# con.moveUntilContact([0, 0, -0.1, 0, 0, 0])
# pos_record('test.npz', con, rec, 30)
# con.moveJ(home_q)


def rmove(con, rec, *, offset: List[float]):
    assert len(offset) == 3
    cur = np.array(rec.getActualTCPPose())
    con.moveL(cur + [*offset, 0, 0, 0])


def rmove_up(con, rec, *, meters: float):
    assert meters > 0
    rmove(con, rec, offset=[0, 0, meters])


def rmove_down(con, rec, *, meters: float):
    assert meters > 0
    rmove(con, rec, offset=[0, 0, -meters])


def move_down_until_contact(con, *, speed: float):
    assert speed > 0
    con.moveUntilContact([0, 0, -speed, 0, 0, 0])


def pick_test(con, rec, gripper):
    teach_move(con, rec)

    # move down, grab, move up
    move_down_until_contact(con, speed=0.1)
    rmove_down(con, rec, meters=0.01)
    gripper.suction(True)
    rmove_up(con, rec, meters=0.1)

    # to target hover
    rmove(con, rec, offset=[0, 0.5, 0])

    # move down, place, go up again
    move_down_until_contact(con, speed=0.1)
    gripper.suction(False)
    rmove_up(con, rec, meters=0.1)


def main():
    con = rtde_control.RTDEControlInterface(UR5_IP)
    rec = rtde_receive.RTDEReceiveInterface(UR5_IP)
    con.setTcp([0, 0, 0.145, 0, 0, 0])
    succ = VacuumGripper(UR5_IP, 63352)

    print('Done with setup. Waiting...', end='')
    # time.sleep(5)
    print('Executing')

    print('NOW MINE')

    pick_test(con, rec, succ)


if __name__ == '__main__':
    main()

# [-1.0923  1.1673 -0.1315 -0.7061  2.9747 -0.0425]

# 0,1 - > (-1.4501,  0.752,  -0.1503) -1.0894  2.8775 -0.1091]
# 1,1 - > (-1.1018,  0.4868, -0.2143) -1.287   2.7572 -0.0968]
# 1,0 -> (-0.7392, 1.0385, -0.1517) -0.4376  3.0709 -0.1174]
# 0,0 -> (-0.8882,  1.288,  -0.0934)  0.1694 -3.1306  0.1155]
