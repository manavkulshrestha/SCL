import numpy as np
import os.path as osp
import open3d as o3d

ddata = osp.abspath('../../rp/data/dep_data')
pdata = osp.abspath('../../rp/data/pcd_data')

example = '0_0.npz'
dep_file = np.load(osp.join(ddata, example))
pcd_file = np.load(osp.join(pdata, example))

pcd, oid, tid = [pcd_file[x] for x in ['pc', 'oid', 'tid']]
nid, depg, nf = [dep_file[x] for x in ['node_ids', 'depg', 'nodes_feat']]

def make_pcd(xyz, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        if len(colors) == 3:
            colors = np.tile(colors, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def typeidx_colors(typeidx):
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


oid_tid_map = dict(zip(oid, tid))

obj_points, obj_colors = [], []

for node, feats in zip(nid, nf):
    feats_reshaped = feats.reshape(-1, 3)
    obj_points.append(feats_reshaped)

    cols = [typeidx_colors(oid_tid_map[node])] * len(feats_reshaped)
    obj_colors.append(cols)

obj_points = np.vstack(obj_points)
obj_colors = np.vstack(obj_colors)

# o3dpcd = make_pcd(pcd, typeidx_colors(tid))
# o3d.visualization.draw_geometries([o3dpcd])
o3dfeats = make_pcd(obj_points, obj_colors)
o3d.visualization.draw_geometries([o3dfeats])