import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# used to find orthogonal vectors
arbitrary_vector = np.array([1, 0, 0])


def playing_with_proj():
    point = np.array([1, 2, 3])
    for n in [-5, 1, 5]:
        normal = np.array([1, 1, n])

        # a plane is a*x+b*y+c*z+d=0
        # [a,b,c] is the normal. Thus, we have to calculate
        # d and we're set
        d = -point.dot(normal)

        # create x,y
        xx, yy = np.meshgrid(range(10), range(10))

        # calculate corresponding z
        zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

        # plot the surface
        ax.plot_surface(xx, yy, zz)

    x, y, z = point
    plt.scatter(x, y, s=z, marker='o', color='red')
    ax.set_xlabel('X_values', fontweight='bold')
    ax.set_ylabel('Y_values', fontweight='bold')
    ax.set_zlabel('Z_values', fontweight='bold')
    ax.set_box_aspect([5, 5, 5])

    plt.show()


def gen_sphere(radius):
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = radius * np.outer(np.sin(theta), np.cos(phi))
    y = radius * np.outer(np.sin(theta), np.sin(phi))
    z = radius * np.outer(np.cos(theta), np.ones_like(phi))

    return ax.plot_wireframe(x, y, z, color='k', linewidth=0.3,
                             rstride=1, cstride=1)


def convert_latlon_to_cartesian(latlon_rad, radius_of_sphere):
    lat, lon = latlon_rad[0], latlon_rad[1]
    # using this formula: https://stackoverflow.com/a/1185413/19589299
    x = radius_of_sphere * np.cos(lat) * np.cos(lon)
    y = radius_of_sphere * np.cos(lat) * np.sin(lon)
    z = radius_of_sphere * np.sin(lat)
    return np.array([x, y, z])


def convert_cartesian_to_latlon_rad(coords_cart, radius_of_sphere):
    lat = np.arcsin(coords_cart[2] / radius_of_sphere)
    lon = np.arctan2(coords_cart[1], coords_cart[0])
    return lat, lon


def gen_square_in_3d(lat_lon_delta, normal, radius_of_sphere):
    # this should generate a planar square perpendicular to given normal w/ given edge length
    # it should return a series of (x,y,z) points that are on the sphere

    # determine lat/long (degrees) of point on surface of normal vect
    magnitude = np.linalg.norm(normal)  # Calculate the magnitude
    normalized_vector = normal / magnitude  # Normalize the vector
    ax.scatter(normalized_vector[0],
               normalized_vector[1], normalized_vector[2])

    # now trying this formula: https://rbrundritt.wordpress.com/2008/10/14/conversion-between-spherical-and-cartesian-coordinates-systems/
    latitude_of_normal_rad, longitude_of_normal_rad = convert_cartesian_to_latlon_rad(
        normalized_vector, radius_of_sphere)

    normal_point = convert_latlon_to_cartesian(
        [latitude_of_normal_rad, longitude_of_normal_rad], radius_of_sphere)
    # find top/bottom of (diamond) by varying the latitude by amount corresponding to squares hypotenuse/2
    top_of_diamond_lat_rad = latitude_of_normal_rad + lat_lon_delta[0]
    top_of_diamond_cart = convert_latlon_to_cartesian(
        [top_of_diamond_lat_rad, longitude_of_normal_rad], radius_of_sphere)
    bottom_of_diamond_lat_rad = latitude_of_normal_rad - lat_lon_delta[0]
    bottom_of_diamond_cart = convert_latlon_to_cartesian(
        [bottom_of_diamond_lat_rad, longitude_of_normal_rad], radius_of_sphere)
    # find R/L of (diamond) by varying the longitude by amount corresponding to squares hypotenuse/2
    r_of_diamond_long_rad = longitude_of_normal_rad + lat_lon_delta[1]
    r_of_diamond_cart = convert_latlon_to_cartesian(
        [latitude_of_normal_rad, r_of_diamond_long_rad], radius_of_sphere)
    l_of_diamond_long_rad = longitude_of_normal_rad - lat_lon_delta[1]
    l_of_diamond_cart = convert_latlon_to_cartesian(
        [latitude_of_normal_rad, l_of_diamond_long_rad], radius_of_sphere)

    return normal_point, np.array([top_of_diamond_cart, l_of_diamond_cart, bottom_of_diamond_cart, r_of_diamond_cart])


def gen_circle_in_3d(num_of_points, radius, normal):
    # this should generate a planar circle @ given normal w/ given edge length
    # it should return a series of (x,y,z) points
    pass


def gen_triangle_in_3d(num_of_points, edge, normal):
    # this should generate a planar circle @ given normal w/ given edge length
    # it should return a series of (x,y,z) points
    pass


def calc_planar_surface_area(coords):
    # given planar (x,y,z) coords,
    # return SA
    # tough because I'm not gauranteed co-planar points
    pass


def calc_spherical_surface_area(coords, radius):
    # given coords in (lat, long) in radians
    # # !!! order of points matter! they must be in a 'ring'; Clockwise will be positive, CC will be
    #  and radius of sphere that shape is on surface of
    # return SA of spherical shape
    area = 0
    num_of_pts = len(coords)
    x1 = coords[num_of_pts-1][0]
    y1 = coords[num_of_pts-1][1]
    for i in range(num_of_pts):
        x2 = coords[i][0]
        y2 = coords[i][1]
        area += (x2 - x1) * (2 + np.sin(y1) + np.sin(y2))
        x1 = x2
        y1 = y2
    area_of_spherical_polygon = (area * radius**2) / 2
    return area_of_spherical_polygon


if __name__ == "__main__":
    fig = plt.figure()
    ax = Axes3D(fig, box_aspect=[1, 1, 1])
    ax.scatter(0, 0, 0, s=50, color='#377d52')
    radius_of_sphere = 1
    normal = np.array([1, 1, 1])
    gen_sphere(radius_of_sphere)
    normal_pt, tangent_square_pts = gen_square_in_3d(
        [np.pi/10, np.pi/5], normal, radius_of_sphere)
    ax.scatter(normal_pt[0], normal_pt[1], normal_pt[2],
               marker='X', s=25, color="red")
    ax.scatter(
        tangent_square_pts[:, 0], tangent_square_pts[:, 1], tangent_square_pts[:, 2], s=10)
    square_points_lat_lon = [convert_cartesian_to_latlon_rad(
        pt, radius_of_sphere) for pt in tangent_square_pts]
    # !!! order of points matter! they must be in a 'ring'; Clockwise will be positive, CC will be
    area_of_spherical_polygon = calc_spherical_surface_area(
        square_points_lat_lon, radius_of_sphere)
    print(area_of_spherical_polygon)
    plt.show()
