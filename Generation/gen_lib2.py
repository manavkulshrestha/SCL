import os
import numpy as np
from numpy.linalg import norm
import trimesh
import pybullet as p
from scipy.spatial.distance import cdist
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
from numpy import cos, sin
import os
import pybullet_data
from numpy import pi
from bidict import bidict
import time

# ASSET_ROOT = os.path.abspath('urdfc')
# ASSET_ROOT = os.path.abspath('../Generation/urdfc')
ASSET_ROOT = os.path.abspath('Generation/urdfc')

obj_types = ['cube', 'cylinder', 'ccuboid', 'scuboid', 'tcuboid', 'roof', 'pyramid', 'cuboid']
obj_primes = [trimesh.load(ASSET_ROOT + f'/{t}/tinker.obj') for t in obj_types]
cube, cylinder, ccuboid, scuboid, tcuboid, roof, pyramid, cuboid = obj_primes

obj_vertcs = set([len(x.vertices) for x in obj_primes])
# [8, 40, 74, 8, 8, 66, 6, 8]
irg_vertc_map = {
    40: 'cylinder',
    74: 'ccuboid',
    66: 'roof',
    6: 'pyramid',
    8: '[regular]'
}

for x in obj_primes:
    x.apply_scale(0.001)


def get_transform(rotq=None, euler=None, rotvec=None, matrix=None, pos=(0, 0, 0)):
    trans = np.eye(4)

    if rotq is not None:
        trans[:-1, :-1] = R.from_quat(rotq).as_matrix()
    elif euler is not None:
        trans[:-1, :-1] = R.from_euler('xyz', euler).as_matrix()
    elif rotvec is not None:
        trans[:-1, :-1] = R.from_rotvec(rotvec).as_matrix()
    elif matrix is not None:
        trans[:-1, :-1] = matrix

    trans[:-1, -1:] = np.array(pos).reshape(-1, 1)

    return trans


def mesh_center(mesh):
    vert_count = len(mesh.vertices)
    center = None
    assert (vert_count in obj_vertcs)

    if irg_vertc_map[vert_count] == 'ccuboid':
        # cm to nearest
        cm = mesh.center_mass
        (center,), *_ = trimesh.proximity.closest_point(mesh, [cm])
    else:
        center = mesh.center_mass

    return np.array(center)


def to_origin(mesh):
    new_offt = mesh_center(mesh)
    mesh.apply_transform(get_transform(pos=-new_offt))


for x in obj_primes:
    to_origin(x)

OM_MAP = {
    'cube': cube,
    'cylinder': cylinder,
    'ccuboid': ccuboid,
    'scuboid': scuboid,
    'tcuboid': tcuboid,
    'roof': roof,
    'pyramid': pyramid,
    'cuboid': cuboid
}
assert (len(OM_MAP) == len(obj_primes))


def make_obj(otype, om=OM_MAP):
    return om[otype].copy()


import copy

pyramid_lom = {
    60: [(0, 0, 0)],
    round(60 / np.sqrt(2)): [(0, 3 * pi / 4, 0), (0, -3 * pi / 4, 0)],
    30: [(pi / 2, 0, 0), (-pi / 2, 0, 0)],
    20: [(0, 0, pi / 2), (0, 3 * pi / 4, pi / 2), (0, 3 * pi / 4, -pi / 2)]
}

pyramidnf_lom = {
    #     60: [(0,0,0)],
    #     round(60/np.sqrt(2)): [(0,3*pi/4,0), (0,-3*pi/4,0)],
    30: [(pi / 2, 0, 0), (-pi / 2, 0, 0)],
    #     20: [(0,0,pi/2), (0,3*pi/4,pi/2), (0,3*pi/4,-pi/2)]
}

cylinder_lom = {
    30: [(0, 0, 0)]
}

cube_lom = {
    30: [(0, 0, 0)]
}

cuboid_lom = {
    30: [(0, 0, 0), (0, pi / 2, pi / 2)],
    60: [(0, pi / 2, 0)]
}

ccuboid_lom = {
    30: [(0, 0, 0), (0, 0, -pi / 2), (0, 0, pi / 2), (0, 0, pi), (0, pi / 2, pi / 2), (0, -pi / 2, pi / 2)],
    # last is smooth face down, second to last is smooth face up like |
    60: [(0, pi / 2, 0), (0, -pi / 2, 0), (pi / 2, 0, pi / 2), (pi / 2, pi, pi / 2)]
    # smooth side up down (first 2) like --
}

scuboid_lom = {
    15: [(0, pi / 2, 0)],
    30: [(0, 0, 0), (0, pi / 2, pi / 2)]
}

tcuboid_lom = {
    15: [(0, 0, pi / 2), (0, pi / 2, pi / 2)],
    30: [(0, 0, 0), (pi / 2, 0, 0)],
    60: [(0, pi / 2, 0), (pi / 2, 0, pi / 2)]
}

roof_lom = {
    30: [(0, 0, pi / 2)],
    40: [(0, 0, 0)]
    # variable lengths like pi/2,0,0 not included for simplicity
}

# maps (obj_string) to (length -> orientations) map
OT_LOM_MAP = {
    'cube': cube_lom,
    'cuboid': cuboid_lom,
    'ccuboid': ccuboid_lom,
    'scuboid': scuboid_lom,
    'tcuboid': tcuboid_lom,
    'pyramid': pyramid_lom,
    'roof': roof_lom,
    'cylinder': cylinder_lom
}

OTNF_LOM_MAP = {
    'cube': cube_lom,
    'cuboid': cuboid_lom,
    'ccuboid': ccuboid_lom,
    'scuboid': scuboid_lom,
    # 'tcuboid': tcuboid_lom,
    # 'pyramid': pyramidnf_lom, # TODO try modified loms laying down
    'cylinder': cylinder_lom
}

POSS_LENS = np.array(list(set().union(*OT_LOM_MAP.values())))

# maps (length) to (list[obj_string] containing orientations with such lengths)
LEN_OT_MAP = {l: [] for l in POSS_LENS}

for l, offerings in LEN_OT_MAP.items():
    for ot, ot_lom in OT_LOM_MAP.items():
        if l in ot_lom:
            offerings.append(ot)

POSS_LENS_NF = np.array(list(set().union(*OTNF_LOM_MAP.values())))
LEN_OTNF_MAP = {l: [] for l in POSS_LENS_NF}

for l, offerings in LEN_OTNF_MAP.items():
    for ot, ot_lom in OTNF_LOM_MAP.items():
        if l in ot_lom:
            offerings.append(ot)


def choice(l, arg=False):
    idx = np.random.choice(len(l))
    return idx if arg else list(l)[idx]


def top_faces_idx(mesh, strict=False):
    facet_n = mesh.facets_normal
    face_n = mesh.face_normals
    up_n = [0, 0, 1] if strict else facet_n[facet_n[:, -1].argmax()]

    return np.all(np.isclose(face_n, up_n), axis=1).nonzero()


def bottom_faces_idx(mesh, strict=False):
    facet_n = mesh.facets_normal
    face_n = mesh.face_normals
    down_n = [0, 0, -1] if strict else facet_n[facet_n[:, -1].argmin()]

    return np.all(np.isclose(face_n, down_n), axis=1).nonzero()


def faces_centroid(mesh, f_idx):
    faces = mesh.faces[f_idx]
    f_vert = mesh.vertices[faces]
    f_cens = f_vert.mean(axis=1)

    return f_cens.mean(axis=0)


def closest_point(set1, set2):
    dists = cdist(set1, set2)
    min_i, min_j = np.unravel_index(dists.argmin(), dists.shape)

    return min_i, min_j


def closest_points_idx(set1, set2, n):
    dists = cdist(set1, set2)
    indices = dists.flatten().argsort()[:n]
    indices = np.unravel_index(indices, dists.shape)

    return indices


def sample_infaces(mesh, faces_idx, num_points):
    face_areas = mesh.area_faces[faces_idx]
    face_probs = face_areas / np.sum(face_areas)

    ar_w = np.zeros_like(mesh.area_faces)
    ar_w[faces_idx] = mesh.area_faces[faces_idx]

    points, faces_idx = mesh.sample(num_points, return_index=True, face_weight=ar_w)

    return points, faces_idx


def fit_plane(points):
    centroid = points.mean(axis=0)
    _, values, vectors = np.linalg.svd(points - centroid)
    normal = vectors[2]
    d = -np.dot(centroid, normal)
    plane = np.append(normal, d)

    return plane


def get_place_plane(mesh1, mesh2, sim_tol=1e-01, n_closest=10, n_samples=100, hook=None):
    m1_maxh = mesh1.vertices[:, :-1].max()
    m2_maxh = mesh2.vertices[:, :-1].max()

    m1_topf_idx = top_faces_idx(mesh1)
    m2_topf_idx = top_faces_idx(mesh2)

    maxh_diff = m1_maxh - m2_maxh
    heights_sim = abs(maxh_diff) < sim_tol

    m1p, m1pf = sample_infaces(mesh1, m1_topf_idx, n_samples)
    m2p, m2pf = sample_infaces(mesh2, m2_topf_idx, n_samples)

    cp_1, cp_2 = closest_points_idx(m1p, m2p, n_closest)
    obj1_p, obj2_p = m1p[cp_1], m2p[cp_2]
    plane_points = np.concatenate([obj1_p, obj2_p])

    # mean point not affected by n on each object
    pp_o1cp = obj1_p.mean(axis=0)
    pp_o2cp = obj2_p.mean(axis=0)
    pp_cen = (pp_o1cp + pp_o2cp) / 2

    pp_eqn = fit_plane(plane_points)
    pp_dis = norm(pp_o1cp - pp_o2cp)

    if hook is not None:
        hook[:] = [pp_o1cp, pp_o2cp]

    return pp_eqn, pp_cen, plane_points, pp_dis, heights_sim


def get_bplace_transform(p_mesh):
    bf_idx = bottom_faces_idx(p_mesh)
    bf_cen = faces_centroid(p_mesh, bf_idx)
    return get_transform(pos=-bf_cen)


def get_align_transform(p_mesh, pp_eqn, pp_cen):
    # transform for bottom face to origin 
    bfc_to_ori = get_bplace_transform(p_mesh)

    # get vectors for transformation
    *normal, _ = pp_eqn
    p_u = normal / norm(normal)
    xy_u = np.array([0, 0, 1])
    rotv = np.cross(xy_u, p_u)
    rotv /= norm(rotv)

    # calculate angle for transformation
    angle = np.arccos(xy_u @ p_u)
    while angle > np.pi / 2:
        angle -= np.pi

    # final transformation
    rotv *= angle
    plane_trans = get_transform(rotvec=rotv, pos=pp_cen)

    return plane_trans @ bfc_to_ori


def pspheres(points, radius=.0005):
    return [trimesh.primitives.Sphere(radius=radius, center=pt) for pt in points]


def get_obj_pose1(col, allow_nonflat=False):
    OTLOMMAP = OT_LOM_MAP
    if not allow_nonflat:
        OTLOMMAP = OTNF_LOM_MAP

    tfc_idx = top_faces_idx(col)
    points = col.vertices[col.faces[tfc_idx].flatten()]
    *abc, d = fit_plane(points)

    # must be relatively flat
    if not np.isclose([0, 0, 1], np.abs(abc), rtol=0, atol=1e-3).all():
        return None

    # select object to place
    oty, lom_map = choice(OTLOMMAP.items())
    orn = choice(choice(lom_map.values()))  # POSSIBLE THAT: placed on very small

    # get transformation
    obj = make_obj(oty)
    tfc_cen = faces_centroid(col, tfc_idx)  # use centre of mass?
    bfo_idx = bottom_faces_idx(obj)  # use centre of mass?
    bfo_cen = faces_centroid(obj, bfo_idx)  # use centre of mass?

    trans = get_transform(pos=tfc_cen - bfo_cen)
    trans[2, -1] += 0.005

    # turn into format for pybullet
    quat = R.from_matrix(trans[:-1, :-1]).as_quat()
    pos = trans[:-1, -1:].flatten()

    # place object
    obj.apply_transform(trans)

    return oty, quat, pos, obj


def get_obj_pose(col1, col2=None, scene_out=None, verbose=False, allow_nonflat=False):
    if col2 is None:
        return get_obj_pose1(col1, allow_nonflat=allow_nonflat)

    OTLOMMAP = OT_LOM_MAP
    POSSLENS = POSS_LENS
    LENOTMAP = LEN_OT_MAP
    if not allow_nonflat:
        #         print('using nf maps')
        OTLOMMAP = OTNF_LOM_MAP
        POSSLENS = POSS_LENS_NF
        LENOTMAP = LEN_OTNF_MAP
    #     else:
    #         print('using normal maps')

    # get plane for placement
    hook = []
    pp_eqn, pp_cen, pp_points, pp_dis, heights_sim = get_place_plane(col1, col2, hook=hook)

    # decide which object to place
    pp_dis *= 1000
    len_diffs = POSSLENS - pp_dis - (THRESHOLD := 2)

    # no object is big enough
    if np.all(len_diffs < 0):
        return None

    # prioritize large objects if heights are similar, else smaller objects
    criteria = np.nanargmax
    if not heights_sim:
        len_diffs[len_diffs < 0] = np.nan
        criteria = np.nanargmin

    # select obj length closest if not heights sim
    closest_len = POSSLENS[criteria(len_diffs)]

    # select object type depending on length
    po_type = np.random.choice(LENOTMAP[closest_len])

    # select desirable orientation for object
    orn_choices = OTLOMMAP[po_type][closest_len]
    po_orn = orn_choices[np.random.choice(len(orn_choices))]

    # get transformation for beginning object pose
    p_obj = make_obj(po_type)
    b_orn = get_transform(euler=po_orn)
    p_obj_copy = p_obj.copy()  # align_trans depends on bottom face being correct
    p_obj_copy.apply_transform(b_orn)

    # get angle of hook line projection on xy with x axis
    plp1, plp2 = np.array(hook)[:, :-1]
    if norm(plp1) < norm(plp2):  # plp1 is further from origin now
        plp1, plp2 = plp2, plp1
    xangle = np.arctan2(*((plp1 - plp2)[::-1]))  # angle with x axis in plane
    alwxtrans = get_transform(euler=(0, 0, xangle))  # angle along z axis

    # get transformation to place object
    align_trans = get_align_transform(p_obj_copy, pp_eqn, pp_cen) @ alwxtrans @ b_orn

    # TESTING WITH MODIFYING Z A BIT TO HELP INTERSECTIONS
    #     print('before')
    #     print(align_trans)
    align_trans[2, -1] += 0.005
    #     print('after')
    #     print(align_trans)

    # get quaternion and position information
    quat = R.from_matrix(align_trans[:-1, :-1]).as_quat()
    pos = align_trans[:-1, -1:].flatten()

    # place object on plane
    align_trans[:-1, :-1] = R.from_quat(quat).as_matrix()
    p_obj.apply_transform(align_trans)

    if scene_out is not None:
        scene_out.add_geometry([col1, col2])
        scene_out.add_geometry(pspheres(pp_points))
        scene_out.add_geometry(pspheres([pp_cen]))
        scene_out.add_geometry([p_obj])

        scene_out.add_geometry(pspheres(hook, radius=.001))

    return po_type, quat, pos, p_obj


class Camera():
    @staticmethod
    def camera_upvec(pos_vec):
        theta = np.arccos(pos_vec[-1])
        sintheta = np.sin(theta)
        phi = np.arccos(pos_vec[0] / sintheta)  # fails if sintheta = 0

        u1 = np.array([cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)])
        # u2 = np.array([-sin(phi), cos(phi), 0])

        return -u1

    def reset_view(self, pos=None, target=None):
        if pos is not None:
            self.pos_vec = np.array(pos)
            self.up_vec = self.camera_upvec(self.pos_vec)

        if target is not None:
            self.target = np.array(target)

        self.viewMat = p.computeViewMatrix(
            cameraEyePosition=self.pos_vec,
            cameraTargetPosition=self.target,
            cameraUpVector=self.up_vec)

    def __init__(self, pos, target=(0, 0, 0), height=512, width=512):
        #         self.pos_vec = np.array(pos)
        #         self.target = np.array(target)
        self.reset_view(pos=pos, target=target)

        #         self.width = 1280
        #         self.height = 720
        self.width = height
        self.height = width

        self.viewMat = p.computeViewMatrix(
            cameraEyePosition=self.pos_vec,
            cameraTargetPosition=self.target,
            cameraUpVector=self.up_vec)

        self.projMat = p.computeProjectionMatrixFOV(
            fov=70.25,
            aspect=1.0,
            nearVal=0.01,
            farVal=3.0)

    def get_image(self, pos=None, target=None):
        self.reset_view(pos=pos, target=target)

        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.viewMat,
            projectionMatrix=self.projMat)

        return rgbImg, depthImg, segImg

    def get_point_cloud(self, pos=None, target=None):
        # get a depth image
        _, depth, seg = self.get_image(pos=pos, target=target)

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.projMat).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.viewMat).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / self.height, -1:1:2 / self.width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixel_idx = z < 0.99
        pixels = pixels[pixel_idx]

        # mine
        seg = seg.reshape(-1)[pixel_idx]
        # end mine

        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3:4]
        points = points[:, :3]

        # assume table is seg[k] = 0
        not_table_idx = seg != 0

        return points[not_table_idx], seg[not_table_idx]


class PBObjectLoader:
    def __init__(self, asset_root):
        self.obj_ids = []
        self.oid_typ_map = {}
        self.obj_poses = {}
        self.asset_root = os.path.abspath(asset_root)

    def load_obj(self, otype, pos=(0, 0, 0), quat=None, euler=None, wait=100, wait_debug=False, slow=False):
        if euler is not None:  # not really idiot proof
            quat = p.getQuaternionFromEuler(euler)
        elif quat is None:
            quat = (0, 0, 0, 1)

        oid = p.loadURDF(os.path.join(self.asset_root, f'{otype}.urdf'), pos, quat)

        if wait_debug:
            input()

        for _ in range(wait):
            p.stepSimulation()
            if slow:
                time.sleep(1. / 240.)

        if wait_debug:
            input()

        self.obj_ids.append(oid)
        self.oid_typ_map[oid] = otype

        return oid

    def recreate(self, oid_subset=None):
        oid_to_mesh = {}

        if oid_subset is None:
            oid_subset = self.obj_ids

        for oid in oid_subset:
            oty = self.oid_typ_map[oid]
            mesh = make_obj(oty)

            m_pos, m_ori = p.getBasePositionAndOrientation(oid)
            mesh.apply_transform(get_transform(rotq=m_ori, pos=m_pos))

            oid_to_mesh[oid] = mesh

        sc = trimesh.Scene(list(oid_to_mesh.values()))

        return bidict(oid_to_mesh), sc

    def get_objs(self):
        return list(zip(self.obj_ids, self.obj_typ))

    def populate_posorns(self):
        for oid in self.obj_ids:
            self.obj_poses[oid] = p.getBasePositionAndOrientation(oid)


PLANE_ROOT = os.path.abspath('urdf')


def swap_objects(obj1, obj2):
    pos1, orn1 = p.getBasePositionAndOrientation(obj1)
    pos2, orn2 = p.getBasePositionAndOrientation(obj2)

    p.resetBasePositionAXndOrientation(obj1, pos2, orn2)
    p.resetBasePositionAndOrientation(obj2, pos1, orn1)


def place_next_to(obj1, obj2, offset):
    pos1, orn1 = p.getBasePositionAndOrientation(obj1)
    pos2, orn2 = p.getBasePositionAndOrientation(obj2)

    p.resetBasePositionAndOrientation(obj1, np.add(pos2, offset), orn1)


def put_in_other_orn(obj1, obj2):
    pos1, orn1 = p.getBasePositionAndOrientation(obj1)
    pos2, orn2 = p.getBasePositionAndOrientation(obj2)

    p.resetBasePositionAndOrientation(obj1, pos1, orn2)


def simulate_scene_pc(cameras, ret_img=False, slow=False, wait=0, num_levels=4, level_bounds=None):
    # loader = PBObjectLoader('urdfc')
    loader = PBObjectLoader('Generation/urdfc')

    otypes = list(OM_MAP.keys())
    otcols = [x for x in otypes if x not in ['roof', 'pyramid', 'tcuboid', 'scuboid']]

    bound = .1

    # maps that level's objects ids to their meshes
    level = [[] for _ in range(num_levels)]

    # TODO
    # maybe cm recreate everytime? # YES, later

    # debug stuff
    coli_sc = None
    pmesh = None
    scp = None
    mo1, mo2, = None, None
    # end debug stuff

    # seed = 7174 or np.random.randint(0, 10000)
    # seed = 348 or np.random.randint(0, 10000)
    COLL_DIST = -1e-3

    level_bounds = list(range(1, num_levels+1))[::-1] if level_bounds is None else level_bounds

    try:
        # level 0
        cm0 = trimesh.collision.CollisionManager()
        attempt = 100
        while attempt > 0 and level_bounds[0] >= len(level[0]):
            # decide candidate state
            c_pos = (*np.random.uniform(-bound, bound, size=2), 0)
            c_typ = choice(otcols)
            c_orn_poss = list(OT_LOM_MAP[c_typ].values())
            c_orn = choice(choice(c_orn_poss))

            # put in state
            c_mesh = OM_MAP[c_typ].copy()
            c_mesh.apply_transform(get_transform(euler=c_orn, pos=c_pos))

            # is far enough
            dist, name = cm0.min_distance_single(c_mesh, return_name=True)
            if dist > bound / 5:
                # add to pybullet and collisionmanager
                o_id = loader.load_obj(c_typ, euler=c_orn, pos=c_pos, wait=wait, slow=slow)
                cm0.add_object(str(o_id), c_mesh)

                # record of which object meshes are in which level
                level[0].append(o_id)
            else:
                attempt -= 1

        # levels 1 onwards
        for cur_level_idx in range(1, num_levels):
            prev_level = level[cur_level_idx-1]
            prev_avail = set(prev_level)
            is_last = cur_level_idx == num_levels-1

            # place on object pairs
            stop_placing = False
            for i, o_id1 in enumerate(prev_level):
                for j, o_id2 in enumerate(prev_level[i + 1:], start=i + 1):
                    if stop_placing:
                        continue

                    # run placement algorithm
                    oim_map, _ = loader.recreate([o_id1, o_id2])
                    mo1, mo2 = list(oim_map.values())

                    object_info = get_obj_pose(mo1, mo2, allow_nonflat=is_last)
                    if object_info is None:
                        continue
                    potype, pquat, ppos, pmesh = object_info

                    # check for collisions with current level and place if none
                    _, coli_sc = loader.recreate()
                    cmi, _ = trimesh.collision.scene_to_collision(coli_sc)

                    if (tdis := cmi.min_distance_single(pmesh)) > 0:
                        o_id = loader.load_obj(potype, quat=pquat, pos=ppos, wait=wait, slow=slow)
                        level[cur_level_idx].append(o_id)
                        prev_avail -= {o_id1, o_id2}

                    stop_placing = len(level[cur_level_idx]) >= level_bounds[cur_level_idx]

            # place single blocks on available spots
            for avail in prev_avail:
                if len(level[cur_level_idx]) >= level_bounds[cur_level_idx]:
                    break

                oim_map, _ = loader.recreate([avail])
                mesh, = list(oim_map.values())

                object_info = get_obj_pose(mesh, allow_nonflat=is_last)
                if object_info is None:
                    continue
                potype, pquat, ppos, pmesh = object_info

                _, coli_sc = loader.recreate()
                cmi, _ = trimesh.collision.scene_to_collision(coli_sc)

                if (tdis := cmi.min_distance_single(pmesh)) > COLL_DIST:
                    o_id = loader.load_obj(potype, quat=pquat, pos=ppos, wait=wait, slow=slow)
                    level[cur_level_idx].append(o_id)

    #     # level 2
    #     #     print('PLACING LEVEL 2')
    #     for i, o_id1 in enumerate(level[1]):
    #         for j, o_id2 in enumerate(level[1][i + 1:], start=i + 1):
    #             # run placement algorithm
    #             oim_map, _ = loader.recreate([o_id1, o_id2])
    #             mo1, mo2 = list(oim_map.values())
    #
    #             object_info = get_obj_pose(mo1, mo2, allow_nonflat=True)
    #             if object_info is None:
    #                 continue
    #
    #             potype, pquat, ppos, pmesh = object_info
    #
    #             # check for collisions with current level and place if none
    #             _, coli_sc = loader.recreate()
    #             cmi, _ = trimesh.collision.scene_to_collision(coli_sc)
    #
    #             if (tdis := cmi.min_distance_single(pmesh)) > COLL_DIST:
    #                 #                 print('SUCCESS!', tdis)
    #                 o_id = loader.load_obj(potype, quat=pquat, pos=ppos, wait=wait, slow=slow)
    #                 level[2].append(o_id)
    #                 level1_avail -= {o_id1, o_id2}
    #     #             else:
    #     #                 print('FAILURE!', tdis)
    #
    #     #     print('PLACING LEVEL 2 - AVAIL')
    #     #     print(level1_avail)
    #     for avail in level1_avail:
    #         oim_map, _ = loader.recreate([avail])
    #         mesh, = list(oim_map.values())
    #
    #         object_info = get_obj_pose(mesh, allow_nonflat=True)
    #         if object_info is None:
    #             continue
    #         potype, pquat, ppos, pmesh = object_info
    #
    #         _, coli_sc = loader.recreate()
    #         cmi, _ = trimesh.collision.scene_to_collision(coli_sc)
    #
    #         if (tdis := cmi.min_distance_single(pmesh)) > COLL_DIST:
    #             #             print('SUCCESS!', tdis)
    #             o_id = loader.load_obj(potype, quat=pquat, pos=ppos, wait=wait, slow=slow)
    #             level[2].append(o_id)
    # #         else:
    # #             print('FAILURE!', tdis)
    #
    #     level2_avail = set(level[2])
    #
    #
    #     # level 2
    #     #     print('PLACING LEVEL 3')
    #     for i, o_id1 in enumerate(level[2]):
    #         for j, o_id2 in enumerate(level[2][i + 1:], start=i + 1):
    #             # run placement algorithm
    #             oim_map, _ = loader.recreate([o_id1, o_id2])
    #             mo1, mo2 = list(oim_map.values())
    #
    #             object_info = get_obj_pose(mo1, mo2, allow_nonflat=True)
    #             if object_info is None:
    #                 continue
    #
    #             potype, pquat, ppos, pmesh = object_info
    #
    #             # check for collisions with current level and place if none
    #             _, coli_sc = loader.recreate()
    #             cmi, _ = trimesh.collision.scene_to_collision(coli_sc)
    #
    #             if (tdis := cmi.min_distance_single(pmesh)) > COLL_DIST:
    #                 #                 print('SUCCESS!', tdis)
    #                 o_id = loader.load_obj(potype, quat=pquat, pos=ppos, wait=wait, slow=slow)
    #                 level[3].append(o_id)
    #                 level2_avail -= {o_id1, o_id2}
    #     #             else:
    #     #                 print('FAILURE!', tdis)
    #
    #     #     print('PLACING LEVEL 3 - AVAIL')
    #     #     print(level1_avail)
    #     for avail in level2_avail:
    #         oim_map, _ = loader.recreate([avail])
    #         mesh, = list(oim_map.values())
    #
    #         object_info = get_obj_pose(mesh, allow_nonflat=True)
    #         if object_info is None:
    #             continue
    #         potype, pquat, ppos, pmesh = object_info
    #
    #         _, coli_sc = loader.recreate()
    #         cmi, _ = trimesh.collision.scene_to_collision(coli_sc)
    #
    #         if (tdis := cmi.min_distance_single(pmesh)) > COLL_DIST:
    #             #             print('SUCCESS!', tdis)
    #             o_id = loader.load_obj(potype, quat=pquat, pos=ppos, wait=wait, slow=slow)
    #             level[3].append(o_id)
    # #         else:
    # #             print('FAILURE!', tdis)

    except Exception as e:
        pass
        # raise e
        # return None

    try:
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1. / 240.)

        debug = False
        while debug:
            p.stepSimulation()
            time.sleep(1/240)

        loader.populate_posorns()
        unpack_hstack = lambda pci, coi: np.hstack([pci, coi.reshape(-1, 1)])
        pcco = np.concatenate([unpack_hstack(*cam.get_point_cloud()) for cam in cameras])

        pc, co = pcco[:, :3], pcco[:, -1]

        if ret_img:
            re = loader, pc, co, [cam.get_image() for cam in cameras]
        else:
            re = loader, pc, co
        return re
    except Exception as e:
        raise e
        # return None


def pc_reg_merge(pcs, cols, rounding=4):
    reg = {}
    for point, colour in zip(pcs, cols):
        reg[tuple(point.round(rounding))] = colour

    return [np.array(list(x)) for x in (reg.keys(), reg.values())]


def oid_typeidx(loader, oids):
    types = ['cube', 'cylinder', 'ccuboid', 'scuboid', 'tcuboid', 'roof', 'pyramid', 'cuboid']
    type_tidx_map = dict(zip(types, range(1, len(types) + 1)))

    # oid -> type -> typidx
    oid_tidx_map = {oid: type_tidx_map[typ] for oid, typ in loader.oid_typ_map.items()}

    return np.array([oid_tidx_map.get(oid, 0) for oid in oids])


def typeidx_colours(typeidx):
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


def dependence(obj1, obj2):
    """
    returns {
         0 if objects are independent
         1 if object 1 depends on object 2

    }
    """
    ctpts = p.getContactPoints(int(obj1), int(obj2))
    if len(ctpts) == 0:
        return 0
    contact_vec_on_2 = ctpts[0][7]

    return 1 if np.sign(contact_vec_on_2)[-1] == 1 else 0


def dep_graph(node_ids):
    """
    returns an adjacency matrix with shape (len(node_ids),len(node_ids)) where {
        1 == depg[i,j] means i depends on j
        0 == depg[i,j] means i does not depend on j
    }

    note: lower/higher refers to the index in the matrix, not the object ids
    """
    n = len(node_ids)
    depg = np.zeros((n, n))

    for i, obj1 in enumerate(node_ids):
        for j, obj2 in enumerate(node_ids):
            depg[i, j] = dependence(obj1, obj2)

    return depg


if __name__ == '__main__':
    seed = 3829  # or np.random.randint(0, 10000)
    np.random.seed(seed)
    print('SEED', seed)

    cam1 = Camera([0.15, 0.15, .2])
    cam2 = Camera([0.15, -0.15, .2])
    cam3 = Camera([0, 0, .3])
    cams = [cam1, cam2, cam3]

    if (loaderpccoimgs := simulate_scene_pc(cams, ret_img=True)) is not None:
        loader, pc, co, imgs = loaderpccoimgs
