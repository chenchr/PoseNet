#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <fstream>

using namespace std;

inline void write_quaternion_2_file(fstream &file, Eigen::Quaterniond &qua){
    file << qua.w() << " " << qua.x() << " " << qua.y() << " " << qua.z() << endl;
}

inline void read_rotation_from_kitti_pose(fstream &file, Eigen::Matrix3d &rot){
    double x, y, z, w;
    for(int i=0; i<3; ++i){
        file >> x >> y >> z >> w;
        rot(i, 0) = x;
        rot(i, 1) = y;
        rot(i, 2) = z;
    }
}

int main(){
    string path_in = "/home/chenchr/Dataset/kitti/poses/00.txt";
    string path_out = "/home/chenchr/qua_eigen.txt";
    fstream file_in(path_in, ios::in);
    fstream file_out(path_out, ios::out); 
    for(int i=0; i<100; ++i){
        Eigen::Matrix3d temp;
        read_rotation_from_kitti_pose(file_in, temp);
        Eigen::Quaterniond qua(temp);
        write_quaternion_2_file(file_out, qua);
    }
    return 0;
}