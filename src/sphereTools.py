import numpy as np
import xarray as xr
from scipy.ndimage import label, generate_binary_structure
from scipy import spatial
from earth_constants import r_E as r_earth

def latlon2xyz(lat, lon, r=1.0):
    
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    return x, y, z

def xyz2latlon(pt):
    
    pt = normVec(pt)

    sin_lat = pt[2]

    if sin_lat == 1:
        lat = np.pi/2
        lon = 0.0
        return lat, lon
    elif sin_lat == -1:
        lat = np.pi/2
        lon = 0.0
        return lat, lon

    cos_lat = np.sqrt(1.0 - sin_lat**2)
   

 
    lat = np.arcsin(pt[2])
    
    cos_lon = pt[0] / cos_lat
    sin_lon = pt[1] / cos_lat

    lon = np.arctan2(sin_lon, cos_lon)

    return lat, lon
    

def getAngleOnSphere(lat1, lon1, lat2, lon2):
    _lat1 = lat1
    _lat2 = lat2
    
    _lon1 = lon1
    _lon2 = lon2

    cosine = (
        np.cos(_lat1) * np.cos(_lat2) * np.cos(_lon1 - _lon2)
        + np.sin(_lat1) * np.sin(_lat2)
    )

    arc = np.arccos(cosine)

    return arc


def getDistOnSphere(lat1, lon1, lat2, lon2, r=1.0):
    return r * getAngleOnSphere(lat1, lon1, lat2, lon2)

def innerProduct(v1, v2):
    return np.sum(v1 * v2)
 
def outerProduct(v1, v2):
    
    v3 = np.zeros((3,), dtype=v1.dtype)

    v3[0] = v1[1] * v2[2] - v1[2] * v2[1]
    v3[1] = v1[2] * v2[0] - v1[0] * v2[2]
    v3[2] = v1[0] * v2[1] - v1[1] * v2[0]
    
    return v3
    
def normVec(v):
    return v / np.sqrt(np.sum(v * v))

def removeComponent(v, v0):
    v0_norm = normVec(v0)
    v_new = v - innerProduct(v, v0_norm) * v0_norm
    return v_new 
    

def getLocalUnitXYZ(lat, lon):
    
    local_unit_z = normVec(np.array(latlon2xyz(lat, lon)))

    local_unit_x = np.array([
        - np.sin(lon),
          np.cos(lon),
          0.0,
    ])

    local_unit_y = outerProduct(local_unit_z, local_unit_x)


    return local_unit_x, local_unit_y, local_unit_z


def constructGreatCircleSegments(lat, lon, tan_vec, dbeta, n, r=1.0, vec_start="half"):
    
    if len(tan_vec) != 3:
        raise Exception("Tangent vector has to be three-dimensional")
    
    p0 = np.array(latlon2xyz(lat, lon, r=1.0))
    unit_y = p0.copy()
  
    # Make sure unit_x and unit_y are orthogonal 
    unit_x = normVec(removeComponent(tan_vec, unit_y))

    if vec_start == "half":
        betas = ( np.arange(n) + 0.5 ) * dbeta
    elif vec_start == "zero":
        betas = np.arange(n) * dbeta
    else:
        raise Exception("Unknown `vec_start` : %s. Allowed `vec_start`: 'zero', 'half'." % (vec_start,))

    seg_latlon_locs  = np.zeros((n, 2),) # lat-lon of the segments
    seg_locs         = np.zeros((n, 3),) # x-y-z of the segments
    seg_tan_vecs     = np.zeros((n, 3),) # x-y-z
   
    for i, beta in enumerate(betas):

        _p = np.sin(beta) * unit_x + np.cos(beta) * unit_y 
        seg_locs[i, :] = _p

        lat, lon = xyz2latlon(_p)
        seg_latlon_locs[i, 0] = lat
        seg_latlon_locs[i, 1] = lon
 
        local_unit_x = np.cos(beta) * unit_x - np.sin(beta) * unit_y
        

        seg_tan_vecs[i, :] = local_unit_x        

   
    seg_locs *= r

    return seg_locs, seg_tan_vecs, seg_latlon_locs 


def getTangentFromTwoPoints(lat1, lon1, lat2, lon2):

    p1 = np.array(latlon2xyz(lat1, lon1))
    p2 = np.array(latlon2xyz(lat2, lon2))
    tan_vec = normVec(removeComponent(p2 - p1, p1))
    
    return tan_vec


def constructGreatCircleSegmentsBetweenTwoPoints(lat1, lon1, lat2, lon2, n, r=1.0):

    total_angle = getAngleOnSphere(lat1, lon1, lat2, lon2)

    angle_inc = total_angle / n
    tan_vec     = getTangentFromTwoPoints(lat1, lon1, lat2, lon2)

    print("tan_vec: ", tan_vec)
    print("angle_inc: %f deg" % (angle_inc * 180/np.pi))

    return constructGreatCircleSegments(lat1, lon1, tan_vec, angle_inc, n, vec_start="half", r=r)
    

    
"""
    `pts` must have the shape of (npts, dim), where dim=2 in AR detection
    
    This algorithm is copied from 
    https://stackoverflow.com/questions/50468643/finding-two-most-far-away-points-in-plot-with-many-points-in-python
"""
def getTheFarthestPtsOnSphere(pts):

    # Looking for the most distant points
    # two points which are fruthest apart will occur as vertices of the convex hulil

    try:
        candidates = pts[spatial.ConvexHull(pts).vertices, :]
    except Exception as e:
        print("Something happen with QhHull: ", str(e))

        candidates = pts

    # get distances between each pair of candidate points
    # dist_mat = spatial.distance_matrix(candidates, candidates)

    dist_mat = np.zeros((len(candidates), len(candidates)))

    for i in range(len(candidates)):
        for j in range(len(candidates)):

            if i >= j:
                dist_mat[i, j] = 0.0
                continue

            dist_mat[i, j] = getDistOnSphere(candidates[i, 0], candidates[i, 1], candidates[j, 0], candidates[j, 1], r=r_earth)

    # get indices of candidates that are furthest apart
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
            
    farthest_pair = ( candidates[i, :], candidates[j, :] )

    return farthest_pair, dist_mat[i, j]


if __name__  == "__main__" :

    print("Loading matplotlib") 
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as transforms
    from matplotlib.dates import DateFormatter
    import matplotlib.ticker as mticker
    print("done")
   

    r = 5.0
    
    # equator
    angles = np.linspace(0, np.pi * 2, 361)
    eq_x = r * np.cos(angles)
    eq_y = r * np.sin(angles)
    eq_z = np.zeros_like(eq_x)

    # 0 lon
    greenich_x = r * np.cos(angles)
    greenich_y = np.zeros_like(greenich_x)
    greenich_z = r * np.sin(angles)


    # Test constructGreatCircleSegmentsBetweenTwoPoints
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))
    ax.set_box_aspect(aspect=(1, 1, 1))
    
    ax.plot(eq_x, eq_y, eq_z, "r--", label="Equator")
    ax.plot(greenich_x, greenich_y, greenich_z, "g--", label="0-lon")
    
    latlons = np.array([
        [0.0,  90.0],  # lat1, lon1
        [30.0, 90.0],  # lat2, lon2
    ],)

    latlons = np.array([
        [20.0, 45.0],  # lat1, lon1
        [50.0, 90.0],  # lat2, lon2
    ],)


    latlon0 = latlons[0, :]
    latlon1 = latlons[1, :]
    
    seg_locs, seg_tan_vecs, seg_latlon_locs = constructGreatCircleSegmentsBetweenTwoPoints(
        lat1=np.radians(latlon0[0]),
        lon1=np.radians(latlon0[1]),
        lat2=np.radians(latlon1[0]),
        lon2=np.radians(latlon1[1]),
        n = 6,
        r = r,
    ) 

    print(seg_locs[:, 0])

    seg_x = seg_locs[:, 0]
    seg_y = seg_locs[:, 1]
    seg_z = seg_locs[:, 2]

    seg_tan_vec_u = seg_tan_vecs[:, 0]
    seg_tan_vec_v = seg_tan_vecs[:, 1]
    seg_tan_vec_w = seg_tan_vecs[:, 2]

    ax.scatter(seg_x, seg_y, seg_z, s = 20, c="red",)
    ax.quiver(seg_x, seg_y, seg_z, seg_tan_vec_u, seg_tan_vec_v, seg_tan_vec_w, color="black")

    ax.scatter([r,], [0, ], [0, ], marker="*", s=100, c="orange", label="(lat, lon) = (0, 0)", zorder=99)
    ax.scatter([0,], [0, ], [r, ], marker="*", s=100, c="blue", label="lat = 90 deg", zorder=99)
 

    for i in range(seg_locs.shape[0]):
        
        rot90_tan_vec = normVec(outerProduct(seg_tan_vecs[i, :], seg_locs[i, :]))

        sub_seg_locs, sub_seg_tan_vecs, sub_seg_latlon_locs = constructGreatCircleSegments(
            lat=seg_latlon_locs[i, 0],
            lon=seg_latlon_locs[i, 1],
            tan_vec = rot90_tan_vec,
            dbeta = np.radians(2),
            n = 20,
            r = r,
            vec_start="zero",
        )
    
        ax.scatter(sub_seg_locs[:, 0], sub_seg_locs[:, 1], sub_seg_locs[:, 2], s = 10, c="magenta",)


    ax.set_title("Test constructGreatCircleSegmentsBetweenTwoPoints")   

    plt.show()



 
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))
    ax.set_box_aspect(aspect=(1, 1, 1))

    ax.plot(eq_x, eq_y, eq_z, "r--", label="Equator")
    ax.plot(greenich_x, greenich_y, greenich_z, "g--", label="0-lon")
    
    latlons = np.array([
        [30.0,  0.0],
        [ 0.0, 90.0],
        [70.0, 20.0],
    ],)

    tan_vecs = np.array([
        [0.0, 1.0, 0.0], 
        [0.0, 0.0, 1.0], 
        [0.5, 1.0, 0.3], 
    ],)


    for i in range(latlons.shape[0]):

        latlon = latlons[i, :]
        tan_vec = tan_vecs[i, :]

        seg_locs, seg_tan_vecs, seg_latlon_locs = constructGreatCircleSegments(
            lat=np.radians(latlon[0]),
            lon=np.radians(latlon[1]),
            tan_vec = tan_vec,
            dbeta = np.radians(15.0),
            n = int(270/15),
            r = r,
        ) 

        seg_x = seg_locs[:, 0]
        seg_y = seg_locs[:, 1]
        seg_z = seg_locs[:, 2]

        seg_tan_vec_u = seg_tan_vecs[:, 0]
        seg_tan_vec_v = seg_tan_vecs[:, 1]
        seg_tan_vec_w = seg_tan_vecs[:, 2]

        ax.scatter(seg_x, seg_y, seg_z, s = 20, c="red",)
        ax.quiver(seg_x, seg_y, seg_z, seg_tan_vec_u, seg_tan_vec_v, seg_tan_vec_w, color="black")
        ax.quiver(seg_locs[0, 0], seg_locs[0, 1], seg_locs[0, 2], tan_vec[0], tan_vec[1], tan_vec[2], color="green")

    ax.scatter([r,], [0, ], [0, ], marker="*", s=100, c="orange", label="(lat, lon) = (0, 0)", zorder=99)
    ax.scatter([0,], [0, ], [r, ], marker="*", s=100, c="blue", label="lat = 90 deg", zorder=99)
    
    ax.legend()

    plt.show()




