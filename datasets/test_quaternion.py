import numpy as np

def rotation_matrix_to_quaternion(rotation):
    print(rotation.shape)
    print("dtype of rotation:{}".format(rotation.dtype))
    trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    s, w, x, y, z = 0.0, 0.0, 0.0, 0.0, 0.0
    print("trace: {}".format(trace))
    print("type of trace:{}".format(type(trace)))
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

def main():
    #read pose
    path = "/home/chenchr/Dataset/kitti/poses/00.txt"
    pose_list = []
    with open(path, 'r') as ff:
        pose_list = ff.readlines()
    pose_list = [line.strip('\n') for line in pose_list]
    temp = []
    for i in pose_list:
        temp.append(np.array(i.split(' ')).reshape(3, 4)[0:3, 0:3].astype(np.float32))
    pose_list = temp
    output_path = '/home/chenchr/qua.txt'
    with open(output_path, 'w') as ff:
        for rotation in pose_list:
            ff.write(np.array2string(rotation_matrix_to_quaternion(rotation), separator=' ').strip('[]')+'\n')

if __name__ == '__main__':
    main()