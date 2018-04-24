import os
import numpy as np
import pandas as pd
import cv2
import argparse

def get_new_csv(path_corr, path_pose):
    df = pd.DataFrame(pd.read_csv(path_corr))
    df_multi = pd.DataFrame(pd.read_csv(path_pose))
    header_multi = df_multi.columns.tolist()
    df_new, filename = [], []
    header_2 = df.columns.tolist()
    for timestamp in df[header_2[0]]:
        line = df_multi.loc[df_multi[header_multi[0]] == timestamp]
        if line.empty:
            continue
        df_new.append(line[header_multi[0:8]])
        filename.append('{}.png'.format(timestamp))
    col_filename = pd.Series(filename)
    df_new = pd.concat(df_new)
    df_new['filename'] = col_filename.values
    return df_new

class Remapper(object):
    def __init__(self, im_size, old_k, distortion, full):
        self.im_size = im_size
        self.old_k = old_k
        self.distortion = distortion
        self.full = full
        self.new_k, temp = cv2.getOptimalNewCameraMatrix(self.old_k, self.distortion, self.im_size, 
                                                         self.full, self.im_size)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.old_k, self.distortion, None, self.new_k, 
                                                           self.im_size, cv2.CV_16SC2)
        
    def get_undistorted_image(self, image):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

parser = argparse.ArgumentParser()
parser.add_argument('--path-old', required=True)
parser.add_argument('--path-new', required=True)
parser.add_argument('--full', type=int, default=1,
                    help='0 for crop mode, 1 for full mode')

def main():
    args = parser.parse_args()
    path_old = args.path_old
    path_new = args.path_new
    subfolder = os.listdir(args.path_old)
    print(subfolder)
    for folder in subfolder:
        temp = os.path.join(path_new, folder, 'image')
        if not os.path.exists(temp):
            os.makedirs(temp)
    old_size, new_size = (752, 480), (752, 480)
    old_k = np.array([[458.654, 0,       367.215],
                    [0,       457.296, 248.375],
                    [0,       0,       1]]).astype(np.float32)
    distortion =  np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]).astype(np.float32)
    rm = Remapper(old_size, old_k, distortion, args.full)
    for folder in a:
        path_corr = os.path.join(path_old, folder, 'mav0/cam0/data.csv')
        path_pose = os.path.join(path_old, folder, 'mav0/state_groundtruth_estimate0/data.csv')
        df_new = get_new_csv(path_corr, path_pose)
        path_out = os.path.join(path_new, folder, 'data.csv')
        df_new.to_csv(path_out, index=None)
        im_name = df_new['filename']
        
        #write intrinsic
        path_intrinsic = os.path.join(path_new, folder, 'k.txt')
        with open(path_intrinsic, 'w') as ff:
            ff.write("fx, fy, cx, cy\n")
            ff.write('{} {} {} {}\n'.format(rm.new_k[0,0], rm.new_k[0,2], rm.new_k[1,1], rm.new_k[1,2]))
            
        for name in im_name:
            path_in = os.path.join(path_old, folder, 'mav0/cam0/data', name)
            path_out = os.path.join(path_new, folder, 'image', name)
            im_in = cv2.imread(path_in, 0)
            im_out = rm.get_undistorted_image(im_in)
            cv2.imwrite(path_out, im_out)

if __name__ == '__main__':
    main()