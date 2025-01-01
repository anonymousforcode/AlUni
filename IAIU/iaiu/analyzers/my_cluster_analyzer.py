from collections import OrderedDict
from iaiu.analyzers.base_analyzer import Analyzer
from iaiu.utils.logger import logger
import torch 
import numpy as np
import os
from iaiu.agents.iaiu.IAIU_model import compute_cosine_similarity
from iaiu.agents.drq.drq_agent import log_frame
# 已知样本数据与聚类中心
# 通过利用cosine相似度度量将数据进行划分类别，得到每一个数据的聚类划分结果
# 取出前5类，每一类取出与聚类中心最相近的前3个样本来展示
# 保存每个数据的类中心
class MyClusterAnalyzer(Analyzer):
    def __init__(
        self,
        agent,
        proto_model,
        pool,
        batch_size=512,
        ndomain = 0,
        log_n = 3
    ):
        self.work = proto_model.cluster_coef
        self.pool = pool
        self.proto_model = proto_model
        self.batch_size = batch_size 
        self.agent = agent
        self.log_n = log_n
        self.ndomain = ndomain

    def analyze(self, epoch):  # 60000
        if self.work > 0 and epoch>0:
            batch = self.pool.analyze_sample(self.batch_size)
            # 经验回放池中打乱顺序
            frames = batch['frames'] # 节选视频里的三帧
            frame_stack = self.agent.frame_stack # 3帧
            C, H, W = frames.shape[2], frames.shape[3], frames.shape[4]
            # channel、height、width （3*84*84）
            cur_frames = frames[:,:frame_stack].reshape(-1, C*frame_stack, H, W)
            # 3通道RGB，3帧图片，每张大小都是84*84
            obs = self.agent.processor(cur_frames) # CNN编码器
            # 原始的三帧的图像1*9*100*100经过CNN编码器得到1*50
            cur_z = self.proto_model.trunk(obs)
            # s->CNN->全连接->层归一化->tanh->潜在状态表征z
            proto_z = self.proto_model.k_proto
            # 聚类中心的大小：(K聚类数*D维度) 128*50
            # zc = c
            z_sim = compute_cosine_similarity(cur_z,proto_z)
            # 为了挑选最相近的样本出来，因此计算z与c之间的余弦相似度
            # z与c之间的相似度，单个batch中每个潜在状态表征与聚类中心之间的余弦相似性
            # 维度：Batch_N*Batch_N

            # 最相似的三帧拿出来
            # 节选前5类中心
            # 3个最近邻
            sim_cluster0 = z_sim[:,0] # 第一列
            _,ind = torch.topk(sim_cluster0,self.log_n)
            # 取最相近的前三个样本
            for i, topi in enumerate(ind):
                log_frame(cur_frames, topi, "c1_top%d"%i, epoch, "visualize/")
                # c1_top0, c1_top1, c1_top2

            sim_cluster1 = z_sim[:,1]
            _,ind = torch.topk(sim_cluster1,self.log_n)
            for i, topi in enumerate(ind):
                log_frame(cur_frames, topi, "c2_top%d"%i, epoch, "visualize/")

            sim_cluster2 = z_sim[:,2]
            _,ind = torch.topk(sim_cluster2,self.log_n)
            for i, topi in enumerate(ind):
                log_frame(cur_frames, topi, "c3_top%d"%i, epoch, "visualize/")

            sim_cluster3 = z_sim[:,3]
            _,ind = torch.topk(sim_cluster3,self.log_n)
            for i, topi in enumerate(ind):
                log_frame(cur_frames, topi, "c4_top%d"%i, epoch, "visualize/")

            sim_cluster4 = z_sim[:,4]
            _,ind = torch.topk(sim_cluster4,self.log_n)
            for i, topi in enumerate(ind):
                log_frame(cur_frames, topi, "c5_top%d"%i, epoch, "visualize/")

class ClusterSaver(Analyzer):
    def __init__(
        self,
        agent,
        proto_model,
        pool,
        batch_size=1024,
        ndomain = 0,
    ):
        self.pool = pool
        self.proto_model = proto_model
        self.batch_size = batch_size 
        self.agent = agent
        self.ndomain = ndomain

    def analyze(self, epoch): 
        batch = self.pool.analyze_sample(self.batch_size)
        # 打乱后的顺序
        frames = batch['frames'] # (B, traj_len, channel, w, h)
        frame_stack = self.agent.frame_stack
        C, H, W = frames.shape[2], frames.shape[3], frames.shape[4]
        cur_frames = frames[:,:frame_stack].reshape(-1, C*frame_stack, H, W)
        # (B, 9, 84, 84)
        obs = self.agent.processor(cur_frames)
        # CNN编码器
        cur_z = self.proto_model.trunk(obs)
        # 全连接+层归一化
        proto_z = self.proto_model.k_proto
        # 聚类中心c，K*Dz，128*50，全连接神经网络的权重作为初始化的聚类中心c
        z_sim = compute_cosine_similarity(cur_z,proto_z)
        # z与c计算cosine相似度
        
        state = batch['states'][:,frame_stack-1].cpu().numpy()
        physics = batch['physics'][:,frame_stack-1].cpu().numpy()
        label = z_sim.argmax(dim=1).cpu().numpy()
        # z_sim：N*K, 表示第i个样本在第k个类上的相似性，值越大越相似
        # 一行行看，取最大的那个值所在的下标，就是属于这一类
        # 举个例子：6个样本，2个聚类中心
        # 样本：
        #  tensor([[ 1.,  1.,  1.,  1.,  1.],
        #         [-1., -2., -3., -4., -5.],
        #         [ 2.,  4.,  6.,  8., 11.],
        #         [-1., -1., -1.,  1., -1.],
        #         [10.,  9.,  8.,  6.,  5.],
        #         [-7., -3., -2.,  4., -5.]])
        # 聚类中心：
        #  tensor([[ 2.,  2.,  6.,  2.,  2.],
        #         [-1., -1., -1.,  5., -1.]])
        # 余弦相似性：
        #  tensor([[ 0.8682,  0.0830],
        #         [-0.7854, -0.2254],
        #         [ 0.7682,  0.2033],
        #         [-0.6202,  0.7474],
        #         [ 0.8562, -0.0212],
        #         [-0.4646,  0.6770]])
        # 样本类别标签： [0 1 0 1 0 1]
        # 与第一类最相近的前2个样本的索引为： tensor([0, 4])
        # 与第二类最相近的前2个样本的索引为： tensor([3, 5])
        label = np.expand_dims(label,axis=1)
        # 扩充维度 [batch_n*1]
        # [[0]
        #  [1]
        #  [2]
        #  [1]
        #  [0]
        #  [1]
        #  [2]]
        dat = np.concatenate([state, physics, label.astype(int)],axis=1)
        # 相当于把状态与类别标签(聚类结果)保存下来
        os.makedirs(os.path.join(logger._snapshot_dir,'cluster'),exist_ok=True)
        logdir = os.path.join(logger._snapshot_dir,'cluster',f"clures_{epoch}.csv")
        np.savetxt(logdir,dat,delimiter=",")
        # 文件名称：cluster
        # 里面有clures_1.csv ----> clures_200.csv这些文件保存样本及其聚类结果