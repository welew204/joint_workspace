"""These were some (failed) attempts at cleaning/prcoessing the path data"""


def denoise_points(mj_path):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(mj_path)

    # returns the points AND the index of each (remaining) point
    filtered_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=5, std_ratio=3.0)

    denoised_points = np.asarray(filtered_cloud.points)

    print(f"Denoisifying removed {len(mj_path)-len(denoised_points)}")

    return denoised_points

# now trying to just do a smooth curve fit...DOESNT WORK


def smooth_path(mj_path, poly_degree):
    x = mj_path[:, 0]
    y = mj_path[:, 1]
    z = mj_path[:, 2]
    coefficients_y = np.polynomial.polynomial.polyfit(
        x, y, poly_degree)
    coefficients_z = np.polynomial.polynomial.polyfit(
        x, z, poly_degree)
    fitted_y = np.polynomial.polynomial.polyval(x, coefficients_y)
    fitted_z = np.polynomial.polynomial.polyval(x, coefficients_z)

    return (x, fitted_y, fitted_z)

# this one DOES work, but not needed :)


def plot_by_quadrant(subplot_id, jt_center, mvj_path_quadrant, sphere=True, planar_tangent=False):
    # WILL NEED TO HAND IN MORE ARRAYs if want to see other points
    # xc, yc, zc = tj_array[:, 0], tj_array[:, 1], tj_array[:, 2]
    # xo, yo, zo = mj_array[:, 0], mj_array[:, 1], mj_array[:, 2]
    color_list = 'bgrcmykw'
    # ax = figure.add_subplot(
    #    4, 2, subplot_id, projection="3d", aspect="equal")

    #
    ax = plt.axes(projection="3d", aspect="equal")
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(0, -1.0)
    ax.set_zlim3d(-0.5, 0.5)
    ax.set_xlabel('X_values', fontweight='bold')
    ax.set_ylabel('Y_values', fontweight='bold')
    ax.set_zlabel('Z_values', fontweight='bold')

    mvj_path = np.array(mvj_path_quadrant)

    # calc centroid according to points in this quadrant

    centroid = find_centroid(mvj_path)
    # compute avg
    avg_displacement = centroid - jt_center
    rad_avg = np.linalg.norm(avg_displacement)

    xn, yn, zn = mvj_path[:, 0], mvj_path[:, 1], mvj_path[:, 2]

    ax.scatter(xn, yn, zn, s=10, marker='o', color=color_list[subplot_id-1])

    if sphere:
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        x = rad_avg * np.outer(np.sin(theta), np.cos(phi)) + jt_center[0]
        y = rad_avg * np.outer(np.sin(theta), np.sin(phi)) + jt_center[1]
        z = rad_avg * np.outer(np.cos(theta),
                               np.ones_like(phi)) + jt_center[2]

        ax.plot_wireframe(x, y, z, color='k', linewidth=0.3,
                          rstride=1, cstride=1)

        # adding some axial planes to see better

        # Create a meshgrid for the x, y, and z coordinates of the planes
        p1x, p1y = np.meshgrid(np.linspace(jt_center[0]-rad_avg, jt_center[0]+rad_avg, 10),
                               np.linspace(jt_center[1]-rad_avg, jt_center[1]+rad_avg, 10))
        p1z = jt_center[2] * np.ones_like(p1x)
        p2x, p2z = np.meshgrid(np.linspace(jt_center[0]-rad_avg, jt_center[0]+rad_avg, 10),
                               np.linspace(jt_center[2]-rad_avg, jt_center[2]+rad_avg, 10))
        p2y = jt_center[1] * np.ones_like(p2x)
        p3y, p3z = np.meshgrid(np.linspace(jt_center[1]-rad_avg, jt_center[1]+rad_avg, 10),
                               np.linspace(jt_center[2]-rad_avg, jt_center[2]+rad_avg, 10))
        p3x = jt_center[0] * np.ones_like(p3y)

        # Create the figure and axes

        # Plot the planes
        ax.plot_surface(p1x, p1y, p1z, alpha=0.2, color='blue')
        ax.plot_surface(p2x, p2y, p2z, alpha=0.2, color='red')
        ax.plot_surface(p3x, p3y, p3z, alpha=0.2, color='green')

    if planar_tangent:
        # using this as guide: https://stackoverflow.com/questions/74111123/how-to-plot-a-plane-perpendicular-to-a-given-normal-and-position-in-3d-in-matplo
        vect = centroid - jt_center
        ax.plot([centroid[0], jt_center[0]], [
                centroid[1], jt_center[1]], zs=[centroid[2], jt_center[2]])
        # perp_vector = np.cross(centroid, vect)

        # try makin the vect a unit vector
        # - find magnitude
        # - divide vector by magnitude
        normal_magnitude = np.linalg.norm(vect)
        u_vect = vect / normal_magnitude
        # ---> this changes the z values (ht) but not enough to be what I'd expect/want (a sq tangent plane)

        # A plane is given by
        # a*x + b*y + c*z + d = 0
        # where (a, b, c) is the normal.
        # If the point (x, y, z) lies on the plane, then solving for d yield:
        # d = -(a*x + b*y + c*z)
        # aka the dot product of the centroid and the normal
        d = -np.sum(centroid * vect)
        d2 = -np.sum(centroid * u_vect)

        # Create a meshgrid:
        delta = 0.2
        xlim = centroid[0] - delta, centroid[0] + delta
        ylim = centroid[1] - delta, centroid[1] + delta
        xx, yy = np.meshgrid(np.linspace(*xlim, 10), np.linspace(*ylim, 10))

        # Solving the equation above for z:
        # z = -(a*x + b*y +d) / c
        zz = -(vect[0] * xx + vect[1] * yy + d) / vect[2]
        print(f"Zone: {subplot_id}")
        print(f"Min z-value: {np.min(zz)}")
        print(f"Max z-value: {np.max(zz)}")
        print(f"Span: {np.max(zz) - np.min(zz)}\n")

        ax.plot_surface(xx, yy, zz, color='orange', alpha=0.5)

        projected_points = plot_projection(vect, centroid, mvj_path)

        pxn, pyn, pzn = projected_points[:,
                                         0], projected_points[:, 1], projected_points[:, 2]

        ax.scatter(pxn, pyn, pzn, marker='x', color='#be03fc')

    ax.scatter(centroid[0], centroid[1], centroid[2], s=50, color='#377d52')

    ax.scatter(0, 0, 0, marker='X', s=50, color='green')
    ax.scatter(
        jt_center[0], jt_center[1], jt_center[2], s=50, color='#f07b2e')

    # can I add a title to each subplot?
    # make 'em EACH a lil bigger
    # fix the tangent planes (more see thru, smaller)
    ax.set_title(f"Quadrant {subplot_id}")

    plt.show()


def plot_projection(plane_normal, centroid, quadrant_path):
    # plane_normal is a vector normal to plane, given by vector (a,b,c)
    # centroid is calculated spot on plane (x,y,z)
    # quadrant_path is the set of points to project

    # p' = p - (n ⋅ p + d) × n

    # norm_mag_sq = plane_normal[0]**2 + plane_normal[1]**2 + plane_normal[2]**2
    # first attempt!
    """ d = -np.sum(centroid * plane_normal)
    k = (d - (plane_normal[0]*centroid[0]) - (plane_normal[1] *
         centroid[1]) - (plane_normal[2]*centroid[2])) / k_denominator """

    proj_path_points = []
    for pt in quadrant_path:
        """ proj_vect = centroid - pt
        distance_away_from_plane = np.dot(proj_vect, plane_normal)
        vect_away_from_plane = distance_away_from_plane*plane_normal
        projection_a = pt - vect_away_from_plane """

        # take 3... have to also find the unit_normal_vector!
        proj_vect = pt - centroid
        normal_magnitude = np.linalg.norm(plane_normal)
        # TODO handle normal_vector == 0?
        unit_normal_vector = plane_normal / normal_magnitude
        distance_away_from_plane_b = np.dot(proj_vect, unit_normal_vector)
        vect_away_from_plane_b = distance_away_from_plane_b*unit_normal_vector
        projection_b = pt - vect_away_from_plane_b

        # print(f"Projection A == {projection_a}")
        # print(f"Projection B == {projection_b}")
        # print(f"...distance === {np.linalg.norm(projection_a - projection_b)} ")

        """
        dot_p = np.dot(proj_vect, plane_normal)
        # 2nd attempt (ChatGPT version)
        projection = pt - dot_p / norm_mag_sq * plane_normal
        """  # proj_x = point_x + ka
        # proj_y = point_y + kb
        # proj_z = point_z + kc
        """p_x = pt[0] + k*plane_normal[0]
        p_y = pt[1] + k*plane_normal[1]
        p_z = pt[2] + k*plane_normal[2]
        proj_pt = np.array([p_x, p_y, p_z])"""
        proj_path_points.append(projection_b)
    return np.array(proj_path_points)


# useless....

def calc_CARs_volume(jt_center, mj_path_array):
    q1_pts = []
    q2_pts = []
    q3_pts = []
    q4_pts = []
    q5_pts = []
    q6_pts = []
    q7_pts = []
    q8_pts = []
    for pt in mj_path_array:
        # determining q1
        if pt[0] > jt_center[0] and pt[1] > jt_center[1] and pt[2] > jt_center[2]:
            q1_pts.append(pt)
        # determining q2
        elif pt[0] < jt_center[0] and pt[1] > jt_center[1] and pt[2] > jt_center[2]:
            q2_pts.append(pt)
        # determining q3
        elif pt[0] < jt_center[0] and pt[1] > jt_center[1] and pt[2] < jt_center[2]:
            q3_pts.append(pt)
        # determining q4
        elif pt[0] > jt_center[0] and pt[1] > jt_center[1] and pt[2] < jt_center[2]:
            q4_pts.append(pt)
        # determining q5
        elif pt[0] > jt_center[0] and pt[1] < jt_center[1] and pt[2] > jt_center[2]:
            q5_pts.append(pt)
        # determining q6
        elif pt[0] < jt_center[0] and pt[1] < jt_center[1] and pt[2] > jt_center[2]:
            q6_pts.append(pt)
        # determining q7
        elif pt[0] < jt_center[0] and pt[1] < jt_center[1] and pt[2] < jt_center[2]:
            q7_pts.append(pt)
        # determining q8
        elif pt[0] > jt_center[0] and pt[1] < jt_center[1] and pt[2] < jt_center[2]:
            q8_pts.append(pt)
    print(q1_pts)


def compute_CARs_hull(target_joint_pose_id, moving_joint_pose_id, real_landmarks, draw=True):
    target_workspace_lms = []
    for frame in real_landmarks:
        # NEED TO handle if point is not in frame (just skip)
        target_joint = frame[target_joint_pose_id][1:]
        target_workspace_lms.append(target_joint)
        moving_bone_end = frame[moving_joint_pose_id][1:]
        target_workspace_lms.append(moving_bone_end)
    points = np.array(target_workspace_lms, dtype=float)
    workspace = ConvexHull(points)
    volume = workspace.volume

    if draw == True:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        # plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 10), facecolor="w")
        ax = plt.axes(projection="3d")
        scatter_plot = ax.scatter3D(x, y, z)
        for simplex in workspace.simplices:
            ax.plot3D(points[simplex, 0], points[simplex, 1],
                      points[simplex, 2], 's-')

        plt.title(f"Workspace of joint {target_joint_pose_id}", fontsize=30)
        ax.set_xlabel('X_values', fontweight='bold')
        ax.set_ylabel('Y_values', fontweight='bold')

        plt.show()

    return volume


def plot_CARs(avg_radius, jt_center, mvj_path_by_quadrant, sphere=True, show_smoothed=False):
    # WILL NEED TO HAND IN MORE ARRAYs if want to see other points
    # xc, yc, zc = tj_array[:, 0], tj_array[:, 1], tj_array[:, 2]
    # xo, yo, zo = mj_array[:, 0], mj_array[:, 1], mj_array[:, 2]
    color_list = 'bgrcmykw'
    fig = plt.figure(figsize=(10, 10), facecolor="w")
    ax = plt.axes(projection="3d", aspect="equal")
    all_mv_points = []
    for q in range(8):
        mvj_path = np.array(mvj_path_by_quadrant[q])
        if len(mvj_path) == 0:
            continue

        xn, yn, zn = mvj_path[:, 0], mvj_path[:, 1], mvj_path[:, 2]

        # scatter_plot = ax.scatter(xo, yo, zo, marker='^')
        ax.scatter(xn, yn, zn, s=10, marker='o', color=color_list[q])
        # scatter_plot = ax.scatter(xc, yc, zc, marker='x')
        if show_smoothed:
            all_mv_points.extend(mvj_path)

    if sphere:
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        x = avg_radius * np.outer(np.sin(theta), np.cos(phi)) + jt_center[0]
        y = avg_radius * np.outer(np.sin(theta), np.sin(phi)) + jt_center[1]
        z = avg_radius * np.outer(np.cos(theta),
                                  np.ones_like(phi)) + jt_center[2]

        ax.plot_wireframe(x, y, z, color='k', linewidth=0.3,
                          rstride=1, cstride=1)

    if show_smoothed:
        all_pts = np.array(all_mv_points)
        fitted_x, fitted_y, fitted_z = smooth_path(all_pts, 15)
        plt.plot(fitted_x, fitted_y, fitted_z, 'r-', label='fitted curve')

    ax.scatter(0, 0, 0, marker='X', s=50, color='green')
    ax.scatter(
        jt_center[0], jt_center[1], jt_center[2], s=50, color='#f07b2e')
    plt.title(
        f"Display of Moving Joint Path around Target Joint", fontsize=30)
    ax.set_xlabel('X_values', fontweight='bold')
    ax.set_ylabel('Y_values', fontweight='bold')
    ax.set_zlabel('Z_values', fontweight='bold')
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(0, -1.0)
    ax.set_zlim3d(-0.5, 0.5)

    # outfile = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/full_CAR_lower_threshold.png"
    outfile = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/full_CAR_higher_threshold.png"
    # plt.savefig(outfile)
    plt.show()
