# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint

from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from mmdet.models.backbones.resnet import BasicBlock
from ..builder import NECKS

from torch.utils.checkpoint import checkpoint


@NECKS.register_module()
class LSSViewTransformer(BaseModule):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """

    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        accelerate=False,
        sid=False,
        collapse_z=True,
        with_cp=False,
        with_depth_from_lidar=False,
    ):
        super(LSSViewTransformer, self).__init__()
        self.with_cp = with_cp
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.sid = sid
        self.frustum = self.create_frustum(grid_config['depth'],
                                           input_size, downsample)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.initial_flag = True
        self.collapse_z = collapse_z
        self.with_depth_from_lidar = with_depth_from_lidar
        if self.with_depth_from_lidar:
            self.lidar_input_net = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, stride=int(2 * self.downsample / 8),
                          padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True))
            out_channels = self.D + self.out_channels
            self.depth_net = nn.Sequential(
                nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, out_channels, 1))

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    # @info 创建形如[D,fH,fW,3]的相机视椎体（四棱锥台）张量frustum，视椎体由D*fH*fW个点组成，即BEV特征图的几何结构
    # 张量的前三维D,fH,fW分别描述了三维视椎体的深度（层数）、高度和宽度，与BEV特征图的形状一致
    # 张量的最后一维描述了视椎体中每个点的三维坐标(v,u,z)，其中v和u表示对应于图像的像素坐标，z表示相机坐标系下点到坐标原点的距离
    # 代码逻辑与BEVFusion中的同名方法几乎完全一致
    # @param depth_cfg 表示深度轴（Z轴）上的网格设置，格式为（start，end，step）。
    # @param input_size 表示输入图像的大小，格式为（高度，宽度）。
    # @param downsample 表示从输入大小到特征大小的下采样因子。
    # @return 返回视锥体，格式为 D x H x W x 3。
    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        # 获得输入图像的大小和下采样因子
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample

        # 构造视椎体中每个点的z坐标。
        # 首先通过arange方法构造出从start到end、步长为step的等差数列（一维），然后通过view方法将其转换为三维张量，
        # 再通过expand方法分别在第2维和第3维上扩展H_feat倍和W_feat倍，将其扩展为与视锥体顶点坐标相同的形状。
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]

        # 如果启用了 self.sid（表示生成稠密深度视图），则对深度值进行重新缩放。
        # 通过计算指数函数的逆运算，将线性间隔的深度值转换为更接近真实深度分布的值。
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
                              torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        
        # 构造视椎体中每个点的x坐标和y坐标，即（v,u），来自图像的像素坐标
        # x和y是形状为[D,fH,fW]的张量，其中D是深度方向上的分辨率，即深度步长
        # x和y由D个完全相同的fH*fW的二维张量堆叠而成，每个二维张量内的元素值分别表示图像宽度和高度方向上的像素坐标。
        # x中的每一行元素都是从0到iW-1之间均匀分布的fW个数值，也就是说对图像像素宽度进行均匀采样，使得x的形状为[D,fH,fW]        
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

        # 把d、x和y沿着最后一个维度拼接起来，获得每个点的相机空间坐标，即(v,u,z)
        # 拼接后frustum的最终形状为[D,fH,fW,3]，其中3表示(v,u,z)
        # stack(tensors, dim)函数用于沿着指定的维度拼接张量，dim=-1表示沿着最后一个维度拼接
        # D x H x W x 3
        return torch.stack((x, y, d), -1)

    # @info 将相机视椎体frustum的像素坐标转换成雷达坐标系并返回
    # 该方法与BEVFusion中的get_geometry方法功能上完全一致，获得视椎体点云坐标（几何特征）
    # @param sensor2ego 表示传感器到车辆坐标系的变换矩阵
    # @param ego2global 表示车辆坐标系到全局坐标系的变换矩阵。
    # @param cam2imgs   表示相机坐标系到图像坐标系的内参矩阵。
    # @param post_rots  表示相机坐标系中的旋转矩阵，经过图像视角增强之后得到。
    # @param post_trans 表示相机坐标系中的平移向量，经过图像视角增强之后得到。
    # @param bda
    def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                       bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        # 首先获取输入参数中的维度信息，其中 B 是批量大小，N 是相机数量。
        B, N, _, _ = sensor2ego.shape

        # 使用图像增强变换矩阵对视椎体进行变换，提高模型的鲁棒性和泛化能力，该矩阵在训练时随机生成，推理时设为单位矩阵
        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # 进行相机到车辆坐标系的转换。
        # 将 points 中的 x 和 y 坐标乘以 z 坐标，然后将 z 坐标保持不变，得到在相机坐标系中的坐标。
        # 然后将 combine 设置为 sensor2ego 的旋转矩阵与相机内参矩阵的逆矩阵相乘，
        # 再将 combine 与 points 相乘，得到在车辆坐标系中的坐标。
        
        # 将视椎体坐标从像素坐标转成相平面坐标
        # points形状是[B,N,D,H,W,3]，转换前第5维是像素坐标（v,u,z），其中z是相机坐标系下点到坐标原点的距离，即深度。
        # 从像素坐标系转到相机坐标系，首先需要将每个点的v和u坐标分别乘以z坐标，z坐标不变。
        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        
        # 将视椎体坐标从相平面坐标转成雷达系（ego系）坐标
        # 首先乘以相机内参矩阵cam2imgs，得到相机系坐标
        # 再乘以相机系到雷达系（ego系）旋转矩阵，加上相机系到雷达系的平移，得到雷达系坐标
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)

        # 然后将其与 bda 矩阵相乘，得到在 bda 坐标系中的坐标。
        points = bda[:, :3, :3].view(B, 1, 1, 1, 1, 3, 3).matmul(
            points.unsqueeze(-1)).squeeze(-1)
        points += bda[:, :3, 3].view(B, 1, 1, 1, 1, 3)
        # 返回点的坐标
        return points

    def init_acceleration_v2(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N_cams, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)

        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    # @info 完成体素池化（即BEV池化），将视椎特征转成BEV特征
    # @param coor 视椎点云坐标（视椎特征feat 像素点一一对应的几何特征），形状为（B，N，D，H，W，3）
    # @param depth 视椎像素的深度分布信息，形状为（B*N,D,H,W）
    # @param feat 视椎特征，形状为（B*N,C,H,W）
    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        
        # 构造BEV特征的形状，即BEV网格：[B,Z,Y,X,C]
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        
        # 根据准备好的视椎特征分布信息，将视椎特征点分配到对应的体素（BEV网格）中，得到BEV特征
        # 为了提高速度使用到了CUDA加速
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    # @info 完成体素池化（BEV池化）的数据准备，计算每个视椎特征点所属的体素（BEV网格），并按照升序排序。
    #       原理和实现上与LSS的BEV柱体池化基本一致，区别在于LSS的BEV网格在高度方向上只有一层，
    #       即BEV柱体池化，而这里的BEV网格在高度方向上有多层（40层）。
    # @param coor 视椎点云坐标（表示与视椎特征像素点一一对应的几何特征），形状为（B，N，D，H，W，3）
    #       对应于LSS中的几何特征geom_feats，基于几何特征，将视椎特征分配到对应的体素（BEV网格）中。
    # @return 点所属的体素的排名，形状为（N_Points）；
    #         在深度空间中点的保留索引，形状为（N_Points）；
    #         在特征空间中点的保留索引，形状为（N_Points）
    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """

        # 取得视椎体坐标的形状，并计算出总的点数
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W

        # 记录每个点在视椎体中的索引，即视椎体中的第几个点
        # 假设总共40个点，则ranks_depth的值为[0,1,2,3,4,5,6,7,8,9,10,11,...,39]
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)
        
        # 按照视椎体的深度方向分层，每一层的点的索引都是从0开始的连续整数
        # 假设总共40个点，每一层10个点，则ranks_feat的值为[0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,...,9,0,1,2,3,4,5,6,7,8,9]
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()

        # 将视椎体坐标转到体素空间，转成形状为[N_Points,3]的张量
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)

        # 为每个点的坐标添加批次索引信息，形成形状为[num_points, 4]的张量，其中最后一列为批次索引。
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # 过滤掉超出体素框范围的点，保留符合条件的点的索引。
        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        
        # 过滤掉超出体素框范围的点，保留符合条件的点的索引
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]

        # 按照BEV视角下的顺序，对视椎点云坐标计算索引，并按照升序排序
        # grid_size是BEV网格在三个维度上的数量，这里是[1024, 1024, 40]，注意最后一个维度不是1
        # 因此，与LSS的一个显著不同在于，这里的BEV网格并不是柱体，而是多层的立体结构
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        # 判断每个索引是否与相邻的下一个索引不同，如果不同则为True，否则为False
        # 例如区间索引ranks_bev为[0, 1, 2, 3, 3, 4, 5, 5, 5, 6]，则kept为[1, 1, 1, 1, 0, 1, 1, 0, 0, 1]
        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]

        # interval_starts生成一个新的间隔起始索引号，记录了kept中所有为True的索引的位置
        # 还是上面的例子，interval_starts为[0, 1, 2, 3, 5, 6, 9]
        # interval_starts实质上记录了落在同一个BEV网格中的一组视椎特征点的起始索引号
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        
        # interval_lengths使用了LSS中的池化累加技巧，与interval_starts一一对应，记录了落在同一个BEV网格中的一组视椎特征点的数量
        # 还是上面的例子，interval_lengths为[1, 1, 1, 2, 1, 3, 1]，表示第0个BEV网格有1个点……第3个BEV网格有2个点……第6个BEV网格有3个点
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1] # 池化累加技巧
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1] # 生成最后一个BEV网格的点数

        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_lidar_coor(*input[1:7])
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    # @info 相机视图转成BEV图的核心方法，将输入特征图转成BEV特征图
    # @param input[0] 骨干网络提取的原始特征图，格式为(B,N,C,H,W)
    # @param input[1:7] 相机内参以及各种变换矩阵
    # @param depth 深度信息，格式为(B*N,D,H,W)
    # @param tran_feat 特征信息，格式为(B*N,C,H,W)
    # @return 返回BEV特征图
    def view_transform_core(self, input, depth, tran_feat):
        B, N, C, H, W = input[0].shape

        # Lift-Splat
        if self.accelerate: 
            # 加速模式
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)

            bev_feat = bev_feat.squeeze(2)
        else:
            # 常规模式
            # 将相机视椎frustum的像素坐标转换成雷达坐标系
            coor = self.get_lidar_coor(*input[1:7])
            # 将视椎点云坐标coor、深度分布概率depth、图像特征tran_feat给到BEVPoolv2中进行池化操作，得到BEV特征图
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))
        return bev_feat, depth

    # @info 将输入特征图转成BEV特征图
    # @param input[0] 骨干网络提取的原始特征图，格式为(B,N,C,H,W)
    # @param input[1:7] 相机内参以及各种变换矩阵
    # @param depth 深度信息，格式为(B*N,D,H,W)
    # @param tran_feat 特征信息，格式为(B*N,C,H,W)
    # @return 返回BEV特征图   
    def view_transform(self, input, depth, tran_feat):
        for shape_id in range(3):
            assert depth.shape[shape_id+1] == self.frustum.shape[shape_id]
        if self.accelerate:
            self.pre_compute(input)
        return self.view_transform_core(input, depth, tran_feat)

    def forward(self, input, depth_from_lidar=None):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)): of (image-view feature, rots, trans,
                intrins, post_rots, post_trans)

        Returns:
            torch.tensor: Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
        """
        x = input[0]                # 将输入数据的第一个元素赋值给变量x
        B, N, C, H, W = x.shape     # 获得x的形状，分别是批量大小，样本数，通道数，高度和宽度
        x = x.view(B * N, C, H, W)  # 将批量和样本维度合并成一个维度
        if self.with_depth_from_lidar:
            assert depth_from_lidar is not None
            if isinstance(depth_from_lidar, list):
                assert len(depth_from_lidar) == 1
                depth_from_lidar = depth_from_lidar[0]
            h_img, w_img = depth_from_lidar.shape[2:]
            depth_from_lidar = depth_from_lidar.view(B * N, 1, h_img, w_img)
            depth_from_lidar = self.lidar_input_net(depth_from_lidar)
            x = torch.cat([x, depth_from_lidar], dim=1)
        if self.with_cp:
            x =checkpoint(self.depth_net, x)
        else:
            x = self.depth_net(x)  # 将x输入到深度网络depth_net中进行处理

        depth_digit = x[:, :self.D, ...]    # 将x的前D个通道赋值给depth_digit，表示深度信息，depth_digit的形状为(B * N, self.D, H, W)
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]    # 将x的第D到第D+out_channels个通道赋值给tran_feat，表示特征信息，tran_feat的形状为(B * N, self.out_channels, H, W)
        depth = depth_digit.softmax(dim=1)  # 在第一个维度上对深度信息进行softmax操作，得到深度概率分布，depth的形状与depth_digit相同，为(B * N, self.D, H, W)。
        return self.view_transform(input, depth, tran_feat)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True,
                 with_cp=False,
                 stereo=False,
                 bias=0.0,
                 aspp_mid_channels=-1):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        depth_conv_input_channels = mid_channels
        downsample = None

        if stereo:
            depth_conv_input_channels += depth_channels
            downsample = nn.Conv2d(depth_conv_input_channels,
                                    mid_channels, 1, 1, 0)
            cost_volumn_net = []
            for stage in range(int(2)):
                cost_volumn_net.extend([
                    nn.Conv2d(depth_channels, depth_channels, kernel_size=3,
                              stride=2, padding=1),
                    nn.BatchNorm2d(depth_channels)])
            self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
            self.bias = bias
        depth_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
                                      downsample=downsample),
                           BasicBlock(mid_channels, mid_channels),
                           BasicBlock(mid_channels, mid_channels)]
        if use_aspp:
            if aspp_mid_channels<0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.with_cp = with_cp
        self.depth_channels = depth_channels

    def gen_grid(self, metas, B, N, D, H, W, hi, wi):
        frustum = metas['frustum']
        points = frustum - metas['post_trans'].view(B, N, 1, 1, 1, 3)
        points = torch.inverse(metas['post_rots']).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)

        rots = metas['k2s_sensor'][:, :, :3, :3].contiguous()
        trans = metas['k2s_sensor'][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas['intrins']))

        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points += trans.view(B, N, 1, 1, 1, 3, 1)
        neg_mask = points[..., 2, 0] < 1e-3
        points = metas['intrins'].view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points = points[..., :2, :] / points[..., 2:3, :]

        points = metas['post_rots'][...,:2,:2].view(B, N, 1, 1, 1, 2, 2).matmul(
            points).squeeze(-1)
        points += metas['post_trans'][...,:2].view(B, N, 1, 1, 1, 2)

        px = points[..., 0] / (wi - 1.0) * 2.0 - 1.0
        py = points[..., 1] / (hi - 1.0) * 2.0 - 1.0
        px[neg_mask] = -2
        py[neg_mask] = -2
        grid = torch.stack([px, py], dim=-1)
        grid = grid.view(B * N, D * H, W, 2)
        return grid

    def calculate_cost_volumn(self, metas):
        prev, curr = metas['cv_feat_list']
        group_size = 4
        _, c, hf, wf = curr.shape
        hi, wi = hf * 4, wf * 4
        B, N, _ = metas['post_trans'].shape
        D, H, W, _ = metas['frustum'].shape
        grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).to(curr.dtype)

        prev = prev.view(B * N, -1, H, W)
        curr = curr.view(B * N, -1, H, W)
        cost_volumn = 0
        # process in group wise to save memory
        for fid in range(curr.shape[1] // group_size):
            prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...]
            wrap_prev = F.grid_sample(prev_curr, grid,
                                      align_corners=True,
                                      padding_mode='zeros')
            curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
            cost_volumn_tmp = curr_tmp.unsqueeze(2) - \
                              wrap_prev.view(B * N, -1, D, H, W)
            cost_volumn_tmp = cost_volumn_tmp.abs().sum(dim=1)
            cost_volumn += cost_volumn_tmp
        if not self.bias == 0:
            invalid = wrap_prev[:, 0, ...].view(B * N, D, H, W) == 0
            cost_volumn[invalid] = cost_volumn[invalid] + self.bias
        cost_volumn = - cost_volumn
        cost_volumn = cost_volumn.softmax(dim=1)
        return cost_volumn

    def forward(self, x, mlp_input, stereo_metas=None):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)

        if not stereo_metas is None:
            if stereo_metas['cv_feat_list'][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas['downsample'])/\
                               stereo_metas['cv_downsample']
                cost_volumn = \
                    torch.zeros((BN, self.depth_channels,
                                 int(H*scale_factor),
                                 int(W*scale_factor))).to(x)
            else:
                with torch.no_grad():
                    cost_volumn = self.calculate_cost_volumn(stereo_metas)
            cost_volumn = self.cost_volumn_net(cost_volumn)
            depth = torch.cat([depth, cost_volumn], dim=1)
        if self.with_cp:
            depth = checkpoint(self.depth_conv, depth)
        else:
            depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)


class DepthAggregation(nn.Module):
    """pixel cloud feature extraction."""

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = checkpoint(self.reduce_conv, x)
        short_cut = x
        x = checkpoint(self.conv, x)
        x = short_cut + x
        x = self.out_conv(x)
        return x


@NECKS.register_module()
class LSSViewTransformerBEVDepth(LSSViewTransformer):

    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(), **kwargs):
        super(LSSViewTransformerBEVDepth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                                  self.out_channels, self.D, **depthnet_cfg)

    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 4, 4).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],], dim=-1)
        sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    def forward(self, input, stereo_metas=None):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input, stereo_metas)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        bev_feat, depth = self.view_transform(input, depth, tran_feat)
        return bev_feat, depth


@NECKS.register_module()
class LSSViewTransformerBEVStereo(LSSViewTransformerBEVDepth):

    def __init__(self,  **kwargs):
        super(LSSViewTransformerBEVStereo, self).__init__(**kwargs)
        self.cv_frustum = self.create_frustum(kwargs['grid_config']['depth'],
                                              kwargs['input_size'],
                                              downsample=4)
