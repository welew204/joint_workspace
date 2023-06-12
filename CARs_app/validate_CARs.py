import json


def check_variance(samples_filepath, target_joint_id, moving_joint_id):
    checkr = {}
    with open(samples_filepath) as samples:
        samples_obj = json.load(samples)
    for i, run in enumerate(samples_obj.keys()):
        target_joint_path = []
        moving_joint_path = []
        for frame in samples_obj[run]:
            tj_vals = frame[target_joint_id][1:]
            target_joint_path.append(tj_vals)
            mj_vals = frame[moving_joint_id][1:]
            moving_joint_path.append(mj_vals)
        checkr[i] = {"target_joint": target_joint_path,
                     "moving_joint": moving_joint_path}

    j = 0
    problem_inds = []
    while j < len(checkr[0]["target_joint"]):
        run1 = checkr[0]
        run2 = checkr[1]
        run3 = checkr[2]
        r1 = run1["target_joint"][j]
        r2 = run2["target_joint"][j]
        r3 = run3["target_joint"][j]
        if r1 != r2 or r1 != r3 or r2 != r3:
            problem_inds.append(j)
        j += 1

    return problem_inds


if __name__ == "__main__":
    check_variance(
        "/Users/williamhbelew/Hacking/ocv_playground/sample_landmarks.json", 12, 14)
