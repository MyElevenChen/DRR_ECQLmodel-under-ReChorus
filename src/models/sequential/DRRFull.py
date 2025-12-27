# -*- coding: utf-8 -*-
"""
DRRFull: 完整 ECQL 实现
作者：ReChorus-DRR 复刻版
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from models.BaseModel import GeneralModel

# ---------- 工具 ----------
class MLP(nn.Module):
    def __init__(self, input_dim, dims, activation='relu', out_activation=None):
        super().__init__()
        layers = []
        last_dim = input_dim
        for i, d in enumerate(dims[:-1]):
            layers += [nn.Linear(last_dim, d), nn.ReLU() if activation == 'relu' else nn.Tanh()]
            last_dim = d
        layers += [nn.Linear(last_dim, dims[-1])]
        if out_activation == 'relu':
            layers += [nn.ReLU()]
        elif out_activation == 'tanh':
            layers += [nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------- 主模型 ----------
class DRRFull(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'hidden_dim', 'tau', 'discount_factor',
                      'actor_lr', 'critic_lr', 'evi_lr', 'lambda_init', 'lambda_min']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--tau', type=float, default=0.01)
        parser.add_argument('--discount_factor', type=float, default=0.9)
        parser.add_argument('--actor_lr', type=float, default=1e-3)
        parser.add_argument('--critic_lr', type=float, default=1e-3)
        parser.add_argument('--evi_lr', type=float, default=1e-3)
        parser.add_argument('--lambda_init', type=float, default=1.0)
        parser.add_argument('--lambda_min', type=float, default=0.1)
        parser.add_argument('--window_len', type=int, default=10)
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.hidden_dim = args.hidden_dim
        self.tau = args.tau
        self.gamma = args.discount_factor
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.evi_lr = args.evi_lr
        self.lambda_init = args.lambda_init
        self.lambda_min = args.lambda_min
        self.window_len = args.window_len
        self.state_size = self.window_len
        self.top_k = 5
        self.num_classes = 5

        # ---------- 网络 ----------
        self._define_params()
        self.apply(self.init_weights)

        # ---------- 优化器 ----------
        self.actor_opt = torch.optim.Adam(self.actor_params, lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic_params, lr=self.critic_lr)
        self.evi_opt = torch.optim.Adam(self.evi_params, lr=self.evi_lr)

        # ---------- 经验池 & 滑动窗口 ----------
        self.replay_buffer = []
        self.buffer_cap = 50000
        self.batch_size = 64
        self.window = deque(maxlen=self.window_len)

        # ---------- λ 退火 ----------
        self.lambda_ = self.lambda_init
        self.anneal_step = 0
        self.anneal_round = 1000  # 每轮样本数

    # ---------- 网络定义 ----------
    def _define_params(self):
        self.user_emb = nn.Embedding(self.user_num, self.emb_size)
        self.item_emb = nn.Embedding(self.item_num, self.emb_size)

        # SSE
        self.sse = nn.GRU(self.emb_size, self.hidden_dim, batch_first=True)
        self.sse_fc = nn.Linear(self.hidden_dim, self.emb_size)

        # Actor
        self.actor = MLP(self.emb_size * 2, [self.hidden_dim, self.emb_size],
                         activation='relu', out_activation='tanh')

        # Critic
        self.critic = MLP(self.emb_size * 3, [self.hidden_dim, 1],
                          activation='relu', out_activation='linear')
        self.target_critic = MLP(self.emb_size * 3, [self.hidden_dim, 1],
                                 activation='relu', out_activation='linear')

        # Evidence
        self.evidence_net = MLP(self.emb_size * 2, [32, self.num_classes],
                                activation='relu', out_activation='relu')

        # 软初始化
        self._soft_update(self.critic, self.target_critic, tau=1.0)

        # 参数分组
        self.actor_params = list(self.user_emb.parameters()) + \
                            list(self.item_emb.parameters()) + \
                            list(self.actor.parameters()) + \
                            list(self.sse.parameters()) + \
                            list(self.sse_fc.parameters())
        self.critic_params = list(self.critic.parameters())
        self.evi_params = list(self.evidence_net.parameters())

    # ---------- 前向（给 BPR 用） ----------
    def forward(self, feed_dict):
        user_id = feed_dict['user_id']  # [B]
        item_id = feed_dict['item_id']  # [B, 1+neg]
        B, neg_num = item_id.shape

        user_v = self.user_emb(user_id)  # [B, dim]
        item_v = self.item_emb(item_id)  # [B, neg_num, dim]

        # 简单 state（训练初期可用，后期用 SSE）
        state = user_v.unsqueeze(1).repeat(1, neg_num, 1)
        inp = torch.cat([state, item_v], dim=-1)  # [B, neg_num, 2*dim]
        action = self.actor(inp.view(B * neg_num, -1)).view(B, neg_num, -1)
        score = (action * item_v).sum(-1)  # [B, neg_num]
        return {'prediction': score}

    # ---------- 真正训练一步 ----------
    def train_step(self, batch):
        # 1. 数据
        user = batch['user_id']
        pos = batch['item_id'][:, 0]
        neg = batch['item_id'][:, 1:]
        rating = batch['rating'] if 'rating' in batch else torch.full_like(pos, 4).float()
        B = user.size(0)

        # 2. 嵌入
        user_v = self.user_emb(user)
        pos_v = self.item_emb(pos)

        # 3. 滑动窗口 & SSE
        if len(self.window) == 0:
            self.window.extend([torch.zeros_like(pos_v[0]) for _ in range(self.window_len)])
        window_tensor = torch.stack(list(self.window)).unsqueeze(0).repeat(B, 1, 1)  # [B, W, dim]
        _, h = self.sse(window_tensor)
        state = self.sse_fc(h.squeeze(0))  # [B, dim]

        # 4. Actor 生成 action
        action_inp = torch.cat([state, pos_v], dim=-1)
        action = self.actor(action_inp)  # [B, dim]

        # 5. Evidence
        evi_inp = torch.cat([action, pos_v], dim=-1)
        evidence = self.evidence_net(evi_inp) + 1  # [B, 5]
        prob = evidence / evidence.sum(dim=-1, keepdim=True)
        rating_pred = (prob * torch.arange(1, 6).to(prob.device)).sum(dim=-1)
        vacuity = self.num_classes / (evidence.sum(dim=-1) + 1e-8)

        # 6. 证据奖励
        reward = rating_pred + self.lambda_ * vacuity
        self._anneal_lambda(B)

        # 7. Critic
        q_inp = torch.cat([action, state, pos_v], dim=-1)
        q_val = self.critic(q_inp)
        with torch.no_grad():
            q_next = self.target_critic(q_inp)  # 简化：未用 next state
            q_target = reward.unsqueeze(1) + self.gamma * q_next
        critic_loss = F.mse_loss(q_val, q_target)

        # 8. Actor（最大化 Q）
        actor_loss = -self.critic(torch.cat([self.actor(action_inp), state, pos_v], dim=-1)).mean()

        # 9. Evidence Loss
        y_true = F.one_hot((rating.long() - 1).clamp(0, 4), num_classes=5).float()
        evi_loss = self.evidential_loss(evidence, y_true)

        # 10. 回传
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.evi_opt.zero_grad()
        evi_loss.backward()
        self.evi_opt.step()

        # 11. 软更新
        self._soft_update(self.critic, self.target_critic, self.tau)

        # 12. 更新滑动窗口（用 pos_v 均值简化）
        self.window.extend(pos_v.detach().unbind(0))

        return {'loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'evi_loss': evi_loss.item(),
                'lambda': self.lambda_}

    # ---------- 工具 ----------
    def _soft_update(self, source, target, tau):
        with torch.no_grad():
            for src, tgt in zip(source.parameters(), target.parameters()):
                tgt.data.copy_(tau * src.data + (1 - tau) * tgt.data)

    def _anneal_lambda(self, B):
        self.anneal_step += B
        ratio = min(1.0, self.anneal_step / self.anneal_round)
        self.lambda_ = self.lambda_init - ratio * (self.lambda_init - self.lambda_min)

    def evidential_loss(self, evidence, y_true):
        alpha = evidence + 1
        S = alpha.sum(dim=-1, keepdim=True)
        prob = alpha / S
        mse = F.mse_loss(prob, y_true, reduction='none')
        var = prob * (1 - prob) / (S + 1)
        return (mse + var).sum(dim=-1).mean()

    # ---------- 评估 ----------
    @torch.no_grad()
    def evaluate_step(self, batch, top_k=10):
        user = batch['user_id']
        items = batch['item_id'][:, :top_k]
        score = self.forward({'user_id': user, 'item_id': items})['prediction']
        label = batch['label'][:, :top_k]
        hits = (score.argsort(descending=True) < label.sum(-1).unsqueeze(1)).float().sum(-1)
        precision = hits / top_k
        return {'precision': precision.mean().item()}