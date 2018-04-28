import numpy as np
import math

#adapt from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
def rotation_matrix_to_quaternion(rotation):
    trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    s, w, x, y, z = 0.0, 0.0, 0.0, 0.0, 0.0
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    qua = np.array([w, x, y, z])
    return qua.astype(np.float32)

#adapt from http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = quaternion
    
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w

    yy = y * y
    yz = y * z
    yw = y * w

    zz = z * z
    zw = z * w

    rotation = np.eye(3).astype(np.float32)
    
    rotation[0,0] = 1 - 2 * (yy + zz)
    rotation[0,1] =     2 * (xy - zw)
    rotation[0,2] =     2 * (xz + yw)

    rotation[1,0] =     2 * (xy + zw)
    rotation[1,1] = 1 - 2 * (xx + zz)
    rotation[1,2] =     2 * (yz - xw)

    rotation[2,0] =     2 * (xz - yw)
    rotation[2,1] =     2 * (yz + xw)
    rotation[2,2] = 1 - 2 * (xx + yy)

    return rotation

def quaternion_to_rotationVec(quaternion):
    w, x, y, z = quaternion
    if w > 1:
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = [i/norm for i in [w, x, y, z]]
    angle = 2 * math.acos(w)
    s = math.sqrt(1-w*w)
    if s > 0.001:
        multi = angle / s
        x, y, z = [i * multi for i in [x, y, z]]
    else:
        x, y, z = [i * angle for i in [x, y, z]]
    vec = np.array([x, y, z])
    return vec.astype(np.float32)


def vector_to_transform(vector):
    transform = np.eye(4).astype(np.float32)
    transform[0:3,3] = vector[0:3]
    transform[0:3, 0:3] = quaternion_to_rotation_matrix(vector[3:])
    return transform

