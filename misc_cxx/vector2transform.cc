#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void vector2transform(vector<double> &vec, Eigen::Matrix4d &t){
    t(0,3) = vec[0];
    t(1,3) = vec[1];
    t(2,3) = vec[2];
    Eigen::Quaterniond temp;
    temp.w() = vec[3];
    temp.x() = vec[4];
    temp.y() = vec[5];
    temp.z() = vec[6];
    t.block(0,0,3,3) = temp.toRotationMatrix();
}

int main(){
    string path_in = "/home/chenchr/ttx.txt";
    fstream file_in(path_in, ios::in);
    vector<vector<double> > all;
    for(int i=0; i<10; ++i){
        double x, y, z, qw, qx, qy, qz;
        char temp;
        file_in >> x >> temp >> y >> temp >> z >> temp >> qw >> temp >> qx >> temp >> qy >> temp >> qz;
        all.push_back({x, y, z, qw, qx, qy, qz});
    }
    vector<Eigen::Matrix4d> all_T;
    for(int i=0; i<10; ++i){
        all_T.push_back(Eigen::Matrix4d::Identity());
        vector2transform(all[i], all_T[i]);
    }
    for(int i=0; i<10; ++i){
        cout << i << ": " << endl;
        for(int j=0; j<7; ++j)
            cout << all[i][j] << " ";
        cout << endl;
        cout << all_T[i] << endl;
    }
    return 0;
}