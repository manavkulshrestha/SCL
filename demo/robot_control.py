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

SLOW_SPEED = 0.03
VSLOW_SPEED = 0.01
MEDI_SPEED = 0.1


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


def teach_move(con, msg='inp: ', wait_after=5):
    con.teachMode()
    inp = input(msg)
    con.endTeachMode()
    time.sleep(wait_after)
    return inp


def teach_move_timed(con, rec, wait_after=5, timeout=5):
    con.teachMode()
    time.sleep(timeout)
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


def rmove(con, rec, *, offset: List[float], speed: float = None):
    assert len(offset) == 3
    cur = np.array(rec.getActualTCPPose())
    if speed is None:
        con.moveL(cur + [*offset, 0, 0, 0])
    else:
        con.moveL(cur + [*offset, 0, 0, 0], speed)


def rmove_up(con, rec, *, meters: float, speed: float = None):
    assert meters > 0
    rmove(con, rec, offset=[0, 0, meters], speed=speed)


def rmove_down(con, rec, *, meters: float, speed: float = None):
    assert meters > 0
    rmove(con, rec, offset=[0, 0, -meters], speed=speed)


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


def init_obj_poses(con, rec, succ, sub_idx=tuple(range(16)), obj_x=np.zeros((16, 6)), obj_q=np.zeros((16, 6))):
    con.moveJ(home_q)
    assert len(obj_x) == len(obj_q) == 16, 'size not 16 objs'

    for i in sub_idx:
        while True:
            inp = teach_move(con, msg=f'[{i}] Attempt pick? (y/n): ')
            if inp in ['y', 'Y', '']:
                move_down_until_contact(con, speed=SLOW_SPEED)
                rmove_down(con, rec, meters=0.01, speed=VSLOW_SPEED)
                succ.suction(True)
                rmove_up(con, rec, meters=0.2)
                rmove_down(con, rec, meters=0.19, speed=MEDI_SPEED)
                move_down_until_contact(con, speed=VSLOW_SPEED)
                time.sleep(2)
                if not succ.was_timeout():
                    succ.suction(False)
                    break

        obj_x[i] = rec.getActualTCPPose()
        obj_q[i] = rec.getActualQ()
        print(f'recorded x = {obj_x[i]}')

    return np.array(obj_x), np.array(obj_q)


def pick(con, rec, succ, pos, use_suction=True):
    con.moveL(pos + [0, 0, 0.02, 0, 0, 0])
    move_down_until_contact(con, speed=VSLOW_SPEED)
    # time.sleep(100000)
    rmove_down(con, rec, meters=0.01)
    time.sleep(1)
    succ.suction(use_suction and True)

    rmove_up(con, rec, meters=0.02, speed=VSLOW_SPEED)


def move_above(con, pos):
    con.moveL(pos + [0, 0, 0.2, 0, 0, 0])


def place(con, rec, succ, pos):
    con.moveL(pos + [0, 0, 0.02, 0, 0, 0])
    move_down_until_contact(con, speed=VSLOW_SPEED)
    rmove_up(con, rec, meters=0.005, speed=VSLOW_SPEED)
    succ.suction(False)
    time.sleep(1)

    rmove_up(con, rec, meters=0.02, speed=VSLOW_SPEED)


def place_on(con, rec, succ, pos):
    con.moveL(pos + [0, 0, 0.1, 0, 0, 0])
    move_down_until_contact(con, speed=VSLOW_SPEED)
    succ.suction(False)
    time.sleep(1)

    rmove_up(con, rec, meters=0.03, speed=VSLOW_SPEED)


def move_object(con, rec, succ, src, dst, use_suction=True):
    move_above(con, src)
    pick(con, rec, succ, src, use_suction=use_suction)
    move_above(con, src)

    move_above(con, dst)
    place(con, rec, succ, dst)
    move_above(con, dst)


def move_object_on(con, rec, succ, src, dst, use_suction=True):
    move_above(con, src)
    pick(con, rec, succ, src, use_suction=use_suction)
    move_above(con, src)

    move_above(con, dst)
    place_on(con, rec, succ, dst)
    move_above(con, dst)


def get_layers():
    layer1 = [9, 14, 10, 15, 7, 11]  # cube, lrcuboid, cylinder, gcuboid, gcuboid, bcuboid
    layer2 = [0, 13, 3]  # ccuboid, cube, ycuboid
    layer3 = [4, 6, 12, 1]  # cylinder, ycuboid, ycuboid, cube
    layer4 = [8, 2, 5]  # ccuboid, cube, cuboid
    layers = [layer1, layer2, layer3, layer4]

    return layers


def test1(con, rec, suc, obj_x, layers):
    tar_x = []
    order = []

    idx = 0
    for layer in layers:
        for obj_idx in layer:
            pos = obj_x[obj_idx]

            move_above(con, pos)
            pick(con, rec, suc, pos)
            move_above(con, pos)

            teach_move(con, msg=f'[{obj_idx}] attempt place? (y/n)')
            move_down_until_contact(con, speed=VSLOW_SPEED)
            rmove_up(con, rec, meters=0.005, speed=VSLOW_SPEED)
            suc.suction(False)
            tar_pos = rec.getActualTCPPose()

            time.sleep(1)
            rmove_up(con, rec, meters=0.03, speed=VSLOW_SPEED)
            rmove_up(con, rec, meters=0.2)

            print(f'obj_x[{obj_idx}] = {tar_pos}')
            tar_x.append(tar_pos)
            order.append(obj_idx)

            np.savez(f'itest_{idx}', tar_x_i=np.array(tar_x), obj_x_i=obj_x[:idx+1])
            idx += 1

    return np.array(order), np.array(tar_x)


def test2(con, rec, suc, obj_x, tar_x, layers):
    idx = 0
    for layer in layers:
        for obj_idx in layer:
            pos = obj_x[obj_idx]
            tar = tar_x[idx]

            move_above(con, pos)
            pick(con, rec, suc, pos)
            move_above(con, pos)

            move_above(con, tar)
            place(con, rec, suc, tar)
            move_above(con, tar)

            idx += 1

    return layers, tar_x


def move_obj_test(con, rec, suc, subset_idx, obj_x):
    for obj_pos in obj_x[subset_idx].reshape(-1, 6):
        print(f'moving obj at {obj_pos}')
        tar_pos = obj_pos + [0, 0.4, 0, 0, 0, 0]
        move_object(con, rec, suc, obj_pos, tar_pos)


def pawses(con, rec, obj_x):
    obj_x = obj_x.reshape(4, 2, 6)
    obj_x[:, :, 2] = np.max(obj_x[:, :, 2])

    con.moveJ(home_q)
    for i, row in enumerate(obj_x):
        for j, ox in enumerate(row):
            move_above(con, ox)
            con.moveL(ox + [0, 0, 0.02, 0, 0, 0])
            move_down_until_contact(con, speed=VSLOW_SPEED)
            obj_x[i, j] = rec.getActualTCPPose()
            rmove_up(con, rec, meters=0.02, speed=VSLOW_SPEED)
            move_above(con, ox)

    np.savez('obj_pos_8.npz', obj_x=obj_x)


def get_layers2():
    layer1 = np.array([12, 4, 11, 3])-1  # glcuboid, cylinder, gcuboid, cube
    layer2 = np.array([6, 2, 9, 1])-1  # lrcuboid, cube, cylinder, ccuboid
    layer3 = np.array([10, 8])-1  # blcuboid, cube
    layer4 = np.array([7, 5])-1  # ccuboid, rlcuboid

    layers = [layer1, layer2, layer3, layer4]
    return layers


def get_layers3():
    layer1 = [5, 2, 4]  # gcuboid, bcuboid, ccuboid
    layer2 = [7, 1]  # cube, cube
    layer3 = [0]  # ccuboid
    layer4 = [6]  # lgcuboid
    layer5 = [3]  # rcuboid

    layers = [layer1, layer2, layer3, layer4, layer5]
    return layers


def main():
    con = rtde_control.RTDEControlInterface(UR5_IP)
    rec = rtde_receive.RTDEReceiveInterface(UR5_IP)
    con.setTcp([0, 0, 0.145, 0, 0, 0])
    suc = VacuumGripper(UR5_IP, 63352)

    print('Done with setup. Waiting...', end='')
    time.sleep(5)
    print('Executing')

    obj_file = np.load('obj_pos_8.npz')
    obj_x = obj_file['obj_x'][:, :2].reshape(-1, 6)

    layers = get_layers3()

    # con.moveJ(home_q)
    # pawses(con, rec, obj_x)

    # order, tar_x = test1(con, rec, suc, obj_x, layers)
    # np.savez(f'test_demo3.npz', obj_x=obj_x, tar_x=tar_x)

    # con.moveJ(up_q)

    con.moveJ(home_q)
    t1_file = np.load('test_demo3.npz')
    tar_x = t1_file['tar_x']
    test2(con, rec, suc, obj_x, tar_x, layers)


if __name__ == '__main__':
    main()


# [-1.0923  1.1673 -0.1315 -0.7061  2.9747 -0.0425]

# 0,1 - > (-1.4501,  0.752,  -0.1503) -1.0894  2.8775 -0.1091]
# 1,1 - > (-1.1018,  0.4868, -0.2143) -1.287   2.7572 -0.0968]
# 1,0 -> (-0.7392, 1.0385, -0.1517) -0.4376  3.0709 -0.1174]
# 0,0 -> (-0.8882,  1.288,  -0.0934)  0.1694 -3.1306  0.1155]
