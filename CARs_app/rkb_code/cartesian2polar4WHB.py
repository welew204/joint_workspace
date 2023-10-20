# cartesian2polar4whb: Pull out 'circle' described by remote joint

import matplotlib.cm as cm
import pylab
import json
import numpy as np
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use('Agg')  # ASSUME no windowing

# https://github.com/welew204/joint_workspace/blob/trunk/CARs_app/CARs_volume.py

# NOT USED....


def convert_cartesian_to_latlon_rad(coords_cart, radius_of_sphere):
    # from this article: https://rbrundritt.wordpress.com/2008/10/14/conversion-between-spherical-and-cartesian-coordinates-systems/
    lat = np.arcsin(coords_cart[2] / radius_of_sphere)
    lon = np.arctan2(coords_cart[1], coords_cart[0])
    return lat, lon


# 3-dim

# https://stackoverflow.com/q/4116658/1079688


# https://stackoverflow.com/a/4116899/1079688
def spherical(xyz):
    spherePts = np.zeros(xyz.shape)

    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    spherePts[:, 0] = np.sqrt(xy + xyz[:, 2]**2)
    # for elevation angle defined from Z-axis down
    spherePts[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])
    spherePts[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])

    return spherePts

# points2area


def polygonArea(X, Y, npoints):

    area = 0
    prev = npoints-1  # create ring
    for idx in range(npoints):
        area += (X[prev]+X[idx]) * (Y[prev]-Y[idx])
        prev = idx

    return area/2


def loadJointFile(inf):

    xlist = []
    ylist = []
    zlist = []
    ptList = []
    jtData = json.load(open(inf))
    for idx in sorted(list(jtData.keys())):
        pt = jtData[idx]
        # Alternative?
        # xlist.append(pt['x'])
        # ylist.append(pt['y'])
        # zlist.append(pt['z'])
        ptList.append([pt['x'], pt['y'], pt['z']])

    # Alternative?
    # shape = 3 X npoints
    # xyz1 = np.array([xlist,ylist,zlist])

    # shape = npoints X 3
    xyz = np.array(ptList)

    return xyz


def plotDim(dim, lbl, outdir):

    f1 = pylab.figure()
    ax1 = pylab.subplot(111)
    pylab.xlabel("t")
    pylab.ylabel('dim')

    ax1.plot(r)

    # https://stackoverflow.com/a/20642478/1079688
    smooth = savgol_filter(dim, 51, 3)  # window size 51, polynomial order 3
    ax1.plot(smooth)

    plotPath = outdir+f'{lbl}-radius.png'
    print('plotRegion: plotting to', plotPath)
    f1.savefig(plotPath)


def plotRegion(elev, azim, lbl, outdir):

    f1 = pylab.figure()
    ax1 = pylab.subplot(111)
    pylab.xlabel("elev")
    pylab.ylabel('azim')
    # window size 51, polynomial order 3
    smooth_azim = savgol_filter(azim, 51, 3)
    # window size 51, polynomial order 3
    smooth_elev = savgol_filter(elev, 51, 3)

    ax1.plot(smooth_elev, smooth_azim)
    pylab.title(lbl)

    plotPath = outdir+f'{lbl}-region.png'
    print('plotRegion: plotting to', plotPath)
    f1.savefig(plotPath)


if __name__ == '__main__':

    dataDir = '/Users/williamhbelew/Hacking/ocv_playground/CARs_app/rkb_code/'

    for jointName in ['hip', 'GH']:

        jtDataFile = dataDir + f'sample_landmarks_normalized_{jointName}.json'

        xyz = loadJointFile(jtDataFile)

        polar = spherical(xyz)
        r = polar[..., 0]
        elev = polar[..., 1]  # theta
        azim = polar[..., 2]  # phi
        npoints = len(elev)

        plotDim(r, jointName+'_r', dataDir)
        plotDim(elev, jointName+'_elev', dataDir)
        plotDim(azim, jointName+'_azim', dataDir)

        area = polygonArea(elev, azim, npoints)
        print(f'Joint={jointName} Area={area}')

        plotRegion(elev, azim, jointName, dataDir)
