# two methods for calculating area of a polygon on a sphere

# OPTION 1: Projection 1
# LIKE THIS (from SO -> https://stackoverflow.com/a/67429413):
# Modules:
import numpy as np
from numpy import arctan2, cos, sin, sqrt, pi, power, append, diff, deg2rad
from pyproj import Geod
# (Geod docs: https://pyproj4.github.io/pyproj/stable/api/geod.html#geod)


# Define WGS84 as CRS:
geod = Geod('+a=6378137 +f=0.0033528106647475126')

# Data for Colorado (no need to close the polygon):
coordinates = np.array([
    [-102.05, 41.0],
    [-102.05, 37.0],
    [-109.05, 37.0],
    [-109.05, 41.0]])
lats = coordinates[:, 1]
lons = coordinates[:, 0]

# Compute:
area, perim = geod.polygon_area_perimeter(lons, lats)

print(abs(area))  # Positive is counterclockwise, the data is clockwise.


# OPTION 2: Integration of bands:
# from SO: https://stackoverflow.com/a/61184491

def polygon_area(lats, lons, radius=6378137):
    """
    Computes area of spherical polygon, assuming spherical Earth. 
    Returns result in ratio of the sphere's area if the radius is specified.
    Otherwise, in the units of provided radius.
    lats and lons are in degrees.
    """

    lats = np.deg2rad(lats)
    lons = np.deg2rad(lons)

    # Line integral based on Green's Theorem, assumes spherical Earth

    # close polygon
    if lats[0] != lats[-1]:
        lats = append(lats, lats[0])
        lons = append(lons, lons[0])

    # colatitudes relative to (0,0)
    a = sin(lats/2)**2 + cos(lats) * sin(lons/2)**2
    colat = 2*arctan2(sqrt(a), sqrt(1-a))

    # azimuths relative to (0,0)
    az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2*pi)

    # Calculate diffs
    # daz = diff(az) % (2*pi)
    daz = diff(az)
    daz = (daz + pi) % (2 * pi) - pi

    deltas = diff(colat)/2
    colat = colat[0:-1]+deltas

    # Perform integral
    integrands = (1-cos(colat)) * daz

    # Integrate
    area = abs(sum(integrands))/(4*pi)

    area = min(area, 1-area)
    if radius is not None:  # return in units of radius
        return area * 4*pi*radius**2
    else:  # return in ratio of sphere total area
        return area


# OPTION 3 - using Girard's Theorem:
# https://stackoverflow.com/a/19398136

d2r = np.pi/180


def greatCircleBearing(lon1, lat1, lon2, lat2):
    dLong = lon1 - lon2

    s = cos(d2r*lat2)*sin(d2r*dLong)
    c = cos(d2r*lat1)*sin(d2r*lat2) - sin(lat1*d2r) * \
        cos(d2r*lat2)*cos(d2r*dLong)

    return np.arctan2(s, c)


N = len(lons)

angles = np.empty(N)
for i in range(N):

    phiB1, phiA, phiB2 = np.roll(lats, i)[:3]
    LB1, LA, LB2 = np.roll(lons, i)[:3]

    # calculate angle with north (eastward)
    beta1 = greatCircleBearing(LA, phiA, LB1, phiB1)
    beta2 = greatCircleBearing(LA, phiA, LB2, phiB2)

    # calculate angle between the polygons and add to angle array
    angles[i] = np.arccos(cos(-beta1)*cos(-beta2) + sin(-beta1)*sin(-beta2))

area = (sum(angles) - (N-2)*np.pi)*R**2
