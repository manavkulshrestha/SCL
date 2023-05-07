import numpy as np
import os.path as osp
import open3d as o3d

def tid_colors(typeidx):
    return np.array([
        [0, 0, 0],
        [0.4160655362053999, 0.13895618220307226, 0.05400398384796701],
        [0.45815538934366873, 0.5622777225161942, 0.12222557471515583],
        [0.5285494304846649, 0.8052616853729326, 0.47328724440755865],
        [0.520059254934221, 0.4733167572634138, 0.5049641813650521],
        [0.2448837131753191, 0.5174992612426157, 0.8959927219176224],
        [0.0859375, 0.9921875, 1.3359375],
        [0.8728815094572616, 0.11715588167789504, 0.9012921785976408],
        [0.8708585184367256, 0.13537291463132384, 0.2942509320637464]
    ])[typeidx]

def make_pcd(xyz, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        if len(colors) == 3:
            colors = np.tile(colors, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize(pcd):
    o3d.visualization.draw_geometries([pcd])

rp_root = osp.abspath('/home/mk/rp')

pcd_root = osp.join(rp_root, 'data/pcd_data/')
dep_root = osp.join(rp_root, 'data/dep_data/')

i = 20

file_name = f'{i // 1000}_{i % 1000}.npz'
pcd_file = np.load(osp.join(pcd_root, file_name))
dep_file = np.load(osp.join(dep_root, file_name))
pcds, o_ids, t_ids = [pcd_file[x] for x in ['pc', 'oid', 'tid']]
node_ids, dep_g = [dep_file[x] for x in ['node_ids', 'depg']]

types = ['cube','cylinder','ccuboid','scuboid','tcuboid','roof','pyramid','cuboid']
oid_tid = dict(zip(o_ids, t_ids))
tid_name = dict(enumerate(types, start=1))
name = lambda oid: tid_name[oid_tid[oid]]

print(dep_g)
for i, r in enumerate(dep_g):
    for j, c in enumerate(r):
        if c:
            nidi = int(node_ids[i])
            nidj = int(node_ids[j])
            print(f'{name(nidi)}[{nidi}] depends on {name(nidj)}[{nidj}]')

visualize(make_pcd(pcds, colors=tid_colors(t_ids)))