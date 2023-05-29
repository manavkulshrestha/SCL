import time
import numpy as np
import rtde_control
import rtde_receive

from typing import List

from demo.VacuumGripper import VacuumGripper


UR5_IP = '192.168.0.33'
s = 0.5
a = 0.3
up_q = np.array([-1, -0.5, 0, -0.5, -0.5, 0]) * np.pi
home_q = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi


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
    inp = input('inp: ')
    con.endTeachMode()
    time.sleep(wait_after)
    return inp

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


def table_offset_test(con, rec, repeat_times=5):
    for _ in range(repeat_times):
        teach_move(con, rec)
        move_down_until_contact(con, speed=0.1)
        print(rec.getActualTCPPose())


def ee_pose(con, rec):
    fk = con.getForwardKinematics(rec.getActualQ(), con.getTCPOffset())
    return fk


def draw_table_correspondences(con, rec, go_home=False):
    con.moveJ(home_q)
    move_down_until_contact(con, speed=0.1)
    print('p0 =', rec.getActualQ())
    print('x0 =', rec.getActualTCPPose())
    input('Next pos?')

    rmove_up(con, rec, meters=0.1)
    print('i01 =', rec.getActualQ())
    rmove(con, rec, offset=[0.2, 0, 0])
    print('i02 =', rec.getActualQ())
    move_down_until_contact(con, speed=0.1)
    print('p1 =', rec.getActualQ())
    print('x1 =', rec.getActualTCPPose())
    input('Next pos?')

    rmove_up(con, rec, meters=0.1)
    print('i11 =', rec.getActualQ())
    rmove(con, rec, offset=[0, -0.2, 0])
    print('i12 =', rec.getActualQ())
    move_down_until_contact(con, speed=0.1)
    print('p2 =', rec.getActualQ())
    print('x2 =', rec.getActualTCPPose())
    input('Up?')

    if go_home:
        rmove_up(con, rec, meters=0.1)
        con.moveJ(up_q)


def init_obj_poses(con, rec):
    con.moveJ(home_q)

    obj_x = []
    obj_q = []
    for _ in range(12):
        inp = teach_move(con, rec)
        if inp == '':
            move_down_until_contact(con, speed=0.01)
        obj_x.append(rec.getActualTCPPose())
        obj_q.append(rec.getActualQ())

    return np.array(obj_x), np.array(obj_q)


def main():
    con = rtde_control.RTDEControlInterface(UR5_IP)
    rec = rtde_receive.RTDEReceiveInterface(UR5_IP)
    con.setTcp([0, 0, 0.145, 0, 0, 0])
    succ = VacuumGripper(UR5_IP, 63352)

    print('Done with setup. Waiting...', end='')
    time.sleep(0)
    print('Executing')

    # obj_x, obj_q = init_obj_poses(con, rec)
    # print(obj_x)
    # print(obj_q)
    # np.savez('obj_pos.npz', obj_x=obj_x, obj_q=obj_q)
    con.moveJ(home_q)


if __name__ == '__main__':
    main()


# [-1.0923  1.1673 -0.1315 -0.7061  2.9747 -0.0425]

# 0,1 - > (-1.4501,  0.752,  -0.1503) -1.0894  2.8775 -0.1091]
# 1,1 - > (-1.1018,  0.4868, -0.2143) -1.287   2.7572 -0.0968]
# 1,0 -> (-0.7392, 1.0385, -0.1517) -0.4376  3.0709 -0.1174]
# 0,0 -> (-0.8882,  1.288,  -0.0934)  0.1694 -3.1306  0.1155]
