from iaiu.processors.base_processor import Processor
from iaiu.torch_modules.mlp import MLP
import iaiu.torch_modules.utils as ptu
from iaiu.utils.logger import logger
from iaiu.agents.ddpg_agent import plot_scatter
from iaiu.agents.drq.drq_agent import log_frame
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

def sinkhorn(scores, eps=0.05, n_iter=3):
    def remove_infs(x):
        m = x[torch.isfinite(x)].max().item()
        x[torch.isinf(x)] = m
        x[x==0] = 1e-38
        return x
    B, K = scores.shape
    scores = scores.view(B*K)
    Q = torch.softmax(-scores/eps, dim=0)
    Q = remove_infs(Q).view(B,K).T
    r, c = ptu.ones(K)/K, ptu.ones(B)/B
    for _ in range(n_iter):
        u = (r/torch.sum(Q, dim=1))
        Q *= remove_infs(u).unsqueeze(1)
        v = (c/torch.sum(Q,dim=0))
        Q *= remove_infs(v).unsqueeze(0)
    bsum = torch.sum(Q,dim=0,keepdim=True)
    output = ( Q / remove_infs(bsum)).T
    assert torch.isnan(output.sum())==False
    return output

def compute_cosine_similarity(e1, e2):
    e1_norm = torch.norm(e1, dim=-1, p=2, keepdim=True)
    e1 = e1 / e1_norm
    e2_norm = torch.norm(e2, dim=-1, p=2, keepdim=True)
    e2 = e2 / e2_norm
    similarity = torch.mm(e1, torch.t(e2))
    return similarity

def compute_l2_similarity(e1, e2, bound=2):
    e1 = e1[:,None,:]
    e2 = e2[None,:,:]
    diff = e1 - e2
    l2_sim = (diff**2).mean(-1)
    l2_sim = torch.clamp(l2_sim, -bound, bound)
    return l2_sim

def compute_z_c_similarity(z, c):
    # ||z-c||^2 (B*K)
    z_sim = torch.cdist(z, c, p=2) ** 2
    return z_sim

def compute_cl_loss(e1, e2, alpha):
    similarity = compute_cosine_similarity(e1, e2)
    similarity = similarity/alpha
    with torch.no_grad():
        pred_prob = torch.softmax(similarity, dim=-1)
        target_prob = ptu.eye(len(similarity))
        accuracy = (pred_prob * target_prob).sum(-1)
        diff = pred_prob-target_prob
    loss = (similarity*diff).sum(-1).mean()
    return loss, pred_prob, accuracy

def KL_z(z1, z2, tau=0.1, eps=1e-12):
    softmax_gap_1 = F.softmax(z1 / tau, dim = -1)
    softmax_gap_2 = F.softmax(z2 / tau, dim = -1)
    loss_kl = softmax_gap_1 * (torch.log(softmax_gap_1 + eps) - torch.log(softmax_gap_2 + eps))
    return loss_kl.mean()

# HSIC----------------------------------------------------------------------
def pairwise_distances(x):
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)

def HSIC(x, y, s_x=1, s_y=1):
    m, _ = x.shape
    K = GaussianKernelMatrix(x,s_x)
    L = GaussianKernelMatrix(y,s_y)
    H = ptu.eye(m) - 1.0/m * ptu.ones((m,m))
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC

# standard orthogonal constraint ----------------------------------------------------------------
def compute_orth(z1, z2):
    N, D = z1.shape
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    c1 = torch.mm(z1.T, z1) / N
    c2 = torch.mm(z2.T, z2) / N
    loss_dec1 = (ptu.eye(D) - c1).pow(2).mean()
    loss_dec2 = (ptu.eye(D) - c2).pow(2).mean()
    loss = 0.5*loss_dec1 + 0.5*loss_dec2
    return loss


class TModelEMA(nn.Module, Processor):
    def __init__(
        self,
        env,
        processor,
        policy,
        forward_layers=[256, 256],
        activation='relu',
        embedding_size=50,
        k=128,
        cluster_coef=1,
        r_seq = 3,
        tau = 0.99,
        beta = 1, # distance = ||z-c||^2 + ||r-rc||^2
        z_coef = 1,
        alpha = 0.1,
        **proto_kwargs
    ):
        nn.Module.__init__(self)
        self.embedding_size = embedding_size
        self.action_size = env.action_shape[0]
        self.trunk = nn.Sequential(
            nn.Linear(processor.output_shape[0], self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.Tanh())
        self.policy = policy
        self.forward_layers = forward_layers  # [256, 256]
        self.activation = activation  # relu
        self.z_coef = z_coef
        self.action_repeat = env.action_repeat # 2/4/8
        self.alpha = alpha
        self.r_seq = r_seq
        self.cluster_coef = cluster_coef
        # here--------------------------------------------------
        self.k = k  # 128
        self.k_proto = nn.Parameter(ptu.zeros(k, self.embedding_size))
        self.k_proto.requires_grad_(True)
        self.has_init = False
        self.proto_r = ptu.zeros(self.k, 1)  # k*1
        self.has_r_init = False
        self.tau = tau  # 0.99
        self.beta = beta
        self.rew_diff = compute_l2_similarity
        # -------------------------------------------------
        input_size = (self.embedding_size + self.action_size)
        self.transition = MLP(
            input_size,
            self.embedding_size,
            hidden_layers=self.forward_layers,
            activation=self.activation)
        self.forward_trunk = nn.Sequential(
            nn.LayerNorm(self.embedding_size),
            nn.Tanh())
    # -----------------------------
    def init_k_proto(self, z_batch):
        from iaiu.torch_modules.utils import device
        k_proto = nn.Linear(self.embedding_size, self.k).weight.data.to(device)
        self.k_proto.data = k_proto

    def ema_proto_r(self, r, Q, batch_size):
        new_r = ptu.zeros(Q.shape[1], 1)
        idx = torch.argmax(Q, dim=1)
        new_r[torch.unique(idx)] = torch.stack([r[idx == i].float().mean() for i in torch.unique(idx)]).unsqueeze(1)
        if self.has_r_init == 0:
            self.proto_r = new_r
            self.has_r_init = True
        else:
            self.proto_r = self.tau * self.proto_r + (1 - self.tau) * new_r
    # ----------------------------
    def _compute_auxiliary_loss(self, next_z, tar_next_z, alpha):
        if self.z_coef>0:
            loss_cl, _, _ = compute_cl_loss(next_z, tar_next_z, alpha)
            loss_kl = KL_z(next_z, tar_next_z, alpha)
            loss_hsic = 0.5 * (HSIC(next_z, next_z) + HSIC(tar_next_z, tar_next_z))
            loss_derr = compute_orth(next_z, tar_next_z)
            h_loss = loss_cl + loss_kl + loss_hsic + loss_derr
        else:
            h_loss = 0
            loss_cl = 0
            loss_kl = 0
            loss_hsic = 0
            loss_derr = 0
        loss = self.z_coef*h_loss
        return loss, h_loss, loss_cl, loss_kl, loss_hsic, loss_derr

    def compute_auxiliary_loss(self, obs, a, r ,next_obs, next_a, 
            n_step=0, log=False, next_frame=None):
        r = r/self.action_repeat
        # ------------
        batch_size = len(obs)  # 128
        # --------------
        cat_obs = torch.cat([obs, next_obs],dim=0)
        cat_z = self.trunk(cat_obs)
        z, next_z = torch.chunk(cat_z, 2)
        # ------------------
        if not self.has_init:
            self.init_k_proto(z)  # 128(K)*50
            self.has_init = True
        # ----------------
        feature = torch.cat([z, a], dim=-1)
        pred_z = self.transition(feature)
        pred_next_z = self.forward_trunk(pred_z)

        # ------------------------------------------------------------
        model_loss, h_loss, loss_cl, loss_kl, loss_hsic, loss_derr = self._compute_auxiliary_loss(
                            pred_next_z, next_z.detach(), self.alpha)

        # ----------------
        if self.cluster_coef > 0:
            cur_z = z
            self.init_k_proto(cur_z)
            z_sim = compute_z_c_similarity(cur_z, self.k_proto)  # ||z-c||^2 (B*K)
            r_sim = self.rew_diff(r, self.proto_r)
            distance = z_sim + self.beta * r_sim
            p_temp = 1.0 / (1.0 + distance)
            P = p_temp / torch.sum(p_temp, dim=-1, keepdim=True)
            with torch.no_grad():
                Q = sinkhorn(-torch.log(P + 1e-38))
                self.ema_proto_r(r, Q, batch_size)
                Q = Q / torch.sum(Q, dim=-1, keepdim=True)
            cluster_loss = -torch.sum(Q * torch.log(P + 1e-38)) / P.shape[0]
        else:
            cluster_loss = 0
        # -------------------
        loss = model_loss + self.cluster_coef*cluster_loss

        if log and n_step%100==0:
            # --------------
            logger.tb_add_scalar("cluster/cluster_loss", cluster_loss, n_step)
            logger.tb_add_histogram("cluster/proto_r", self.proto_r, n_step)
            # -----------------
            logger.tb_add_scalar("model/cl_loss", loss_cl, n_step)
            logger.tb_add_scalar("model/Orth_loss", loss_derr, n_step)
            logger.tb_add_scalar("model/HSIC_loss", loss_hsic, n_step)
            logger.tb_add_scalar("model/kl_loss", loss_kl, n_step)
        return loss

    def process(self):
        raise NotImplementedError
