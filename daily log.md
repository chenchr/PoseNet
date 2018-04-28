# logging

## 0419

- [x] 测试函数 ```convert_rotation_matrix_to_quaternion```: 跟eigen对比ok

- [x] 在 kitti 00序列上尝试，假设现在都是没有bug的，现在很快loss就降到很低，初步认为车在行走时基本上帧图像的旋转和平移都是差不多的，这样相当于训练样本都没有多样性，现在打算找个别的数据集试一试(euroc)


## 0420

- [x] euroc全部序列内参和畸变参数都是一样，首先要做畸变矫正，把全部数据保存多一份，image_0每个序列大概有六七百张图片的pose需要插值的，现在暂时不管这些。得到了full和crop的去畸变数据集

- [x] 写一下euroc的dataloader

- [x] 整理一下代码框架，使得可以适应各个数据集，尽量不用手动 TODO，改成输入 train seq 和 test seq 的形式

## 0421

- [x] 完整测试下整个框架，特别是旋转和平移部分: ```rotation_matrix_to_quaternion```, ```quaternion_to_rotation_matrix```, ```vector_to_transform``` ok； 注意numpy等*是element-wise的乘法，矩阵乘法要用.dot()...，图片normalize部分出了个错误，[128,128,128]但是有6个channel，所以只有前三个normalize了。。

- [ ] 需要可视化出数据集中旋转、平移的分布，想一下怎么搞，在一个球上？ 平移可以，euroc的平移分布明显比kitti的均匀，旋转不知如何表示

## 0423

- [x] 尝试改变网络，使用cnngeometric_pytorch， 发现几个模型下四元数的误差很快就很低，平移的误差卡在1.2处降不下去，要再检查下代码，想想有什么验证方式，这个最低也只有1.1

- [x] 配置下服务器的环境

- [x] tvnet改成pytorch版本 基本完成，直接跑结果错误，需要显示中间结果debug下

## 0424

- [ ] 加入 data augumentation 

- [x] 看别的sfm网络对kitti数据集是如何训练的，是一个一个序列来还是全部一起，其他的都是多个序列一起的，现在改成
    ```
    train_seq = ['V2_01_easy', 'MH_02_easy', 'V1_03_difficult', 'V1_01_easy', 'V1_02_medium', 'V2_02_medium','MH_04_difficult', 'MH_03_medium'] 

    test_seq  = ['MH_05_difficult', 'V2_03_difficult']
    ```

## 0425

- [x] kitti的平移基本都是朝前，在 kitti 上训练网络 train test 都能收敛，在 euroc 上不行，现在将算 loss 时的 normalize 只在 test 模式下做，train 直接算loss

## 0426

- [x] tvnet 学习一下 tensorflow summary 的使用，逐步对照输出结果: 已经找出问题原因，通过逐步画图 heat map 对比比较好找，发现是pytorch的mask index的bug：例如有时候
    ```
    masks1 = a > b
    c[masks1] = d
    ```
    这样没法正常工作，但是masks1的值都是对的，通过以下代码迂回解决：
    ```
    c = masks1.float() * d
    ```
    后面还要继续搜下为什么

    实验发现求光流前先做高斯平滑对结果影响挺大的

- [ ] 思路1：类似 Direct Multichannel Tracking，利用cnn得到 feature pyramid，然后进行 TV-L1 ，由于整个过程是端到端的，可以直接进行学习
    问题: vectorial 的 TV-L1 好像还没有一个比较简单的可以利用 pytorch 的实现，这部分到时问下老师

- [x] 现在需要先将 TV-L1 的代码整理完整，可以实现训练，已经完成

## 0427

- demon pose 网络的输出是 r，t，scale，其中r的参数化方式是旋转向量，t的 groundtruth 是单位向量，相当于只有方向，然后 loss 是直接拿网络输出和 groundtruth 做差得到 L2 norm 的

- [x] 把旋转参数化从四元数改成旋转向量

- [ ] 实现下 demon 的 pose 估计