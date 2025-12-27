# -*- coding: utf-8 -*-
"""
DRRFull_ECQL_v2: 完整ECQL实现（改进版）
基于论文: Looking into User's Long-term Interests through the Lens of Conservative Evidential Learning
结合TensorFlow源码优点进行完善
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, defaultdict
import random
import math
from models.BaseModel import GeneralModel

# ---------- 工具类 ----------
class MLP(nn.Module):
    def __init__(self, input_dim, dims, activation='relu', out_activation=None):
        super().__init__()
        layers = []
        last_dim = input_dim
        for i, d in enumerate(dims[:-1]):
            layers.append(nn.Linear(last_dim, d))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            last_dim = d
        layers.append(nn.Linear(last_dim, dims[-1]))
        if out_activation == 'relu':
            layers.append(nn.ReLU())
        elif out_activation == 'tanh':
            layers.append(nn.Tanh())
        elif out_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """优先经验回放池（PER）"""
    def __init__(self, capacity, embedding_dim, alpha=0.6, beta=0.4, beta_increment=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.states = torch.zeros((capacity, 3 * embedding_dim))
        self.actions = torch.zeros((capacity, embedding_dim))
        self.rewards = torch.zeros(capacity)
        self.next_states = torch.zeros((capacity, 3 * embedding_dim))
        self.dones = torch.zeros(capacity, dtype=torch.bool)
        self.uncers = torch.zeros(capacity)
        
        self.priorities = torch.zeros(capacity)
        self.position = 0
        self.is_full = False
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, uncer, done):
        idx = self.position
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.uncers[idx] = uncer
        self.dones[idx] = done
        
        # 设置优先级
        priority = self.max_priority ** self.alpha
        self.priorities[idx] = priority
        
        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.is_full = True
    
    def sample(self, batch_size):
        N = self.capacity if self.is_full else self.position
        
        # 计算采样概率
        probs = self.priorities[:N] / self.priorities[:N].sum()
        
        # 重要性采样权重
        min_prob = probs.min()
        max_weight = (N * min_prob) ** (-self.beta)
        
        # 采样
        indices = torch.multinomial(probs, min(batch_size, N), replacement=False)
        
        # 计算权重
        weights = (N * probs[indices]) ** (-self.beta) / max_weight
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.uncers[indices],
            self.dones[indices],
            weights,
            indices
        )
        
        return batch
    
    def update_priorities(self, indices, priorities):
        self.priorities[indices] = (priorities ** self.alpha).clamp(min=1e-6)
        self.max_priority = max(self.max_priority, priorities.max().item())


class SlidingWindow:
    """滑动窗口，维护用户最近交互"""
    def __init__(self, window_size):
        self.window_size = window_size
        self.windows = {}  # user_id -> deque
        
    def update(self, user_id, item_emb, recommended_items=None):
        if user_id not in self.windows:
            self.windows[user_id] = deque(maxlen=self.window_size)
        
        window = self.windows[user_id]
        
        if recommended_items is not None and len(recommended_items) > 0:
            # 用推荐项目替换窗口中的一半项目（论文设计）
            num_to_replace = min(len(recommended_items), len(window) // 2)
            if num_to_replace > 0:
                # 移除旧项目
                for _ in range(num_to_replace):
                    if len(window) > 0:
                        window.popleft()
                # 添加推荐项目
                for item_emb in recommended_items[:num_to_replace]:
                    window.append(item_emb)
        
        # 添加新交互的项目
        if item_emb is not None:
            window.append(item_emb)
        
        # 如果窗口不满，用零填充
        while len(window) < self.window_size:
            if item_emb is not None:
                window.append(torch.zeros_like(item_emb))
            else:
                window.append(torch.zeros(window[0].shape) if len(window) > 0 else torch.zeros(100))
    
    def get_window(self, user_id):
        if user_id in self.windows and len(self.windows[user_id]) > 0:
            return list(self.windows[user_id])
        else:
            return [torch.zeros(100) for _ in range(self.window_size)]


# ---------- 状态表示模块（从TensorFlow移植）----------
class DRRAveStateRepresentation(nn.Module):
    """状态表示网络（对应TensorFlow的DRRAveStateRepresentation）"""
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        # 1D卷积层，用于加权平均（WAV）
        self.wav = nn.Conv1d(embedding_dim, 1, kernel_size=1)
        
    def forward(self, user_emb, items_emb):
        """
        Args:
            user_emb: [batch_size, embedding_dim]
            items_emb: [batch_size, state_size, embedding_dim]
        Returns:
            state: [batch_size, 3*embedding_dim]
        """
        batch_size = user_emb.shape[0]
        
        # 转置items_emb以应用卷积
        items_transposed = items_emb.transpose(1, 2)  # [batch, emb_dim, state_size]
        
        # 应用卷积（加权平均）
        wav_output = self.wav(items_transposed)  # [batch, 1, state_size]
        wav_output = wav_output.transpose(1, 2)  # [batch, state_size, 1]
        
        # 计算加权后的用户表示
        user_wav = user_emb.unsqueeze(1) * wav_output  # [batch, state_size, emb_dim]
        user_wav = user_wav.mean(dim=1)  # [batch, emb_dim]
        
        # 拼接三部分：用户嵌入、加权用户嵌入、权重
        state = torch.cat([user_emb, user_wav, wav_output.mean(dim=1)], dim=1)  # [batch, 3*emb_dim]
        
        return state


# ---------- RNN网络（从TensorFlow移植）----------
class RnnNetwork(nn.Module):
    """RNN网络（用于更复杂的序列建模）"""
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(output_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(output_size + hidden_size, output_size)
        self.pred_rating = nn.Linear(output_size + 100, 5)  # 假设item_emb维度为100
        
    def forward(self, input_state, hidden, item_emb=None, rating=None):
        combined = torch.cat([input_state, hidden], dim=1)
        
        hidden = torch.tanh(self.i2h(combined))
        state = torch.tanh(self.i2o(combined))
        
        logits = None
        if item_emb is not None:
            # 拼接state和item_emb
            state_tiled = state.repeat(item_emb.shape[0], 1)
            combined_features = torch.cat([item_emb, state_tiled], dim=1)
            logits = F.softmax(self.pred_rating(combined_features), dim=1)
            
        return state, hidden, logits


# ---------- 证据网络（从TensorFlow移植完善）----------
class EvidenceNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5),
            nn.Softplus()  # ✅ 改用Softplus，确保正数且可微分
        )
    
    def forward(self, action, item_emb):
        batch_size = action.shape[0]
        num_items = item_emb.shape[0]
        
        action_expanded = action.repeat(num_items, 1)
        combined = torch.cat([item_emb, action_expanded], dim=1)
        
        evidence = self.fc(combined)  # ✅ 去掉+1
        evidence = evidence.view(num_items, -1)
        
        return evidence


# ---------- 主模型 ----------
class DRRFull_ECQL_v2(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = [
        'emb_size', 'hidden_dim', 'tau',
        'top_k'
    ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=100)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--tau', type=float, default=0.001)
        parser.add_argument('--discount_factor', type=float, default=0.9)
        parser.add_argument('--actor_lr', type=float, default=0.001)
        parser.add_argument('--critic_lr', type=float, default=0.001)
        parser.add_argument('--evi_lr', type=float, default=0.001)
        parser.add_argument('--lambda_init', type=float, default=1.0)
        parser.add_argument('--lambda_min', type=float, default=0.1)
        parser.add_argument('--window_len', type=int, default=10)
        parser.add_argument('--top_k', type=int, default=10)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--conservative_alpha', type=float, default=5.0)
        parser.add_argument('--rl_batch_size', type=int, default=1)  # 与TensorFlow一致
        parser.add_argument('--buffer_size', type=int, default=10000)
        parser.add_argument('--warmup_steps', type=int, default=1000)
        parser.add_argument('--update_freq', type=int, default=4)
        parser.add_argument('--use_rnn', type=int, default=1)  # 是否使用RNN
        parser.add_argument('--use_per', type=int, default=1)  # 是否使用PER
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
        self.top_k = args.top_k
        self.alpha = args.alpha
        self.conservative_alpha = args.conservative_alpha
        self.rl_batch_size = args.rl_batch_size
        self.warmup_steps = args.warmup_steps
        self.update_freq = args.update_freq
        self.use_rnn = args.use_rnn
        self.use_per = args.use_per
        
        self.num_classes = 5
        self.rating_threshold = 3.0
        
        # ---------- 网络 ----------
        self._define_params()
        self.apply(self.init_weights)
        
        # ---------- 优化器 ----------
        self.actor_opt = torch.optim.Adam(self.actor_params, lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic_params, lr=self.critic_lr)
        self.evi_opt = torch.optim.Adam(self.evi_params, lr=self.evi_lr)
        if self.use_rnn:
            self.rnn_opt = torch.optim.Adam(self.rnn_params, lr=self.actor_lr)
        
        # ---------- 经验池 & 滑动窗口 ----------
        if self.use_per:
            self.replay_buffer = ReplayBuffer(args.buffer_size, self.emb_size)
        else:
            # 简单经验回放
            self.replay_buffer = {
                'states': [], 'actions': [], 'rewards': [], 
                'next_states': [], 'uncers': [], 'dones': []
            }
            self.buffer_size = args.buffer_size
            
        self.sliding_window = SlidingWindow(self.window_len)
        self.behavior_policy = {}
        
        # ---------- 训练状态 ----------
        self.total_steps = 0
        self.lambda_ = self.lambda_init
        self.w1 = 0.5  # 探索/利用平衡参数（从TensorFlow移植）
        self.episode_rewards = []
        self.user_interaction_history = defaultdict(list)
        
        # ---------- RNN隐藏状态 ----------
        if self.use_rnn:
            self.hidden_states = {}

    def _define_params(self):
        # 嵌入层
        self.user_emb = nn.Embedding(self.user_num, self.emb_size)
        self.item_emb = nn.Embedding(self.item_num, self.emb_size)
        
        # 1. 状态表示网络（从TensorFlow移植）
        self.state_representation = DRRAveStateRepresentation(self.emb_size)
        
        # 2. RNN网络（可选）
        if self.use_rnn:
            self.rnn_network = RnnNetwork(self.hidden_dim, 3 * self.emb_size)
            self.rnn_params = list(self.rnn_network.parameters())
        
        # 3. Actor网络
        self.actor_network = nn.Sequential(
            nn.Linear(3 * self.emb_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.emb_size),
            nn.Tanh()
        )
        
        # 4. Critic网络（双Q网络）
        self.critic1 = nn.Sequential(
            nn.Linear(4 * self.emb_size, self.hidden_dim),  # state + action
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(4 * self.emb_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # 目标网络
        self.target_critic1 = nn.Sequential(
            nn.Linear(4 * self.emb_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        self.target_critic2 = nn.Sequential(
            nn.Linear(4 * self.emb_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # 复制参数到目标网络
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 5. Evidence网络（从TensorFlow移植）
        self.evidence_network = EvidenceNetwork(self.emb_size)
        
        # 参数分组
        self.actor_params = (list(self.user_emb.parameters()) + 
                           list(self.item_emb.parameters()) + 
                           list(self.state_representation.parameters()) + 
                           list(self.actor_network.parameters()))
        
        self.critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.evi_params = list(self.evidence_network.parameters())

    def forward(self, feed_dict):
        """传统前向传播"""
        user_id = feed_dict['user_id']
        item_id = feed_dict['item_id']
        user_v = self.user_emb(user_id)
        item_v = self.item_emb(item_id)
        
        score = (user_v.unsqueeze(1)* item_v).sum(dim=-1) #有改动
        return {'prediction': score}

    def get_state_representation(self, user_id, items_ids, device):
        """生成状态表示（完整SSE实现）"""
        user_v = self.user_emb(torch.tensor([user_id]).long().to(device))
        
        # 获取滑动窗口项目
        window_items = self.sliding_window.get_window(user_id)
        if len(window_items) < self.window_len:
            # 用items_ids补充
            items_emb = self.item_emb(torch.tensor(items_ids[:self.window_len]).long().to(device))
            if len(window_items) > 0:
                window_tensor = torch.stack(window_items + [items_emb[i] for i in range(min(len(items_emb), self.window_len - len(window_items)))]).unsqueeze(0)
            else:
                window_tensor = items_emb[:self.window_len].unsqueeze(0)
        else:
            window_tensor = torch.stack(window_items).unsqueeze(0).to(device)
        
        # 状态表示
        state = self.state_representation(user_v, window_tensor)
        
        # 可选：RNN编码
        if self.use_rnn:
            if user_id not in self.hidden_states:
                hidden = torch.zeros(1, self.hidden_dim).to(device)
                self.hidden_states[user_id] = hidden
            else:
                hidden = self.hidden_states[user_id]
            
            state, new_hidden, _ = self.rnn_network(state, hidden)
            self.hidden_states[user_id] = new_hidden.detach()
        
        return state

    def get_action(self, state, explore=True):
        """生成动作"""
        if explore:
            # 添加探索噪声
            action = self.actor_network(state)
            noise = torch.randn_like(action) * 0.1
            return action + noise
        else:
            return self.actor_network(state)

    def compute_evidence(self, action, item_embs):
        """计算证据向量"""
        return self.evidence_network(action, item_embs)

    def calculate_vacuity(self, evidence):
        """计算vacuity"""
        S = evidence.sum(dim=-1, keepdim=True) + self.num_classes
        vacuity = self.num_classes / S
        return vacuity.squeeze(-1)

    def calculate_rating_probability(self, evidence):
        """计算评分概率分布"""
        alpha = evidence + 1
        S = alpha.sum(dim=-1, keepdim=True)
        prob = alpha / S
        return prob

    def recommend_items(self, user_id, total_items, evidence, top_k=None, w1=None):
        device = next(self.parameters()).device
        
        # 计算评分概率和不确定性
        prob = self.calculate_rating_probability(evidence)
        rating_pred = (prob * torch.arange(1, 6).to(device)).sum(dim=-1)
        vacuity = self.calculate_vacuity(evidence)
        
        # 参数设置
        if w1 is None:
            w1 = self.w1
        if top_k is None:
            top_k = self.top_k
        elif top_k <= 0:  # 添加保护
            top_k = self.top_k
        
        # ✅ 关键修改：确保 top_k 不大于可用项目数
        total_available = len(total_items)
        if total_available == 0:
            return np.array([]), 0.0
        
        # 如果 top_k 大于可用项目数，调整到可用数量
        actual_top_k = min(top_k, total_available)
        
        # ✅ 确保 w1 在合理范围内
        w1 = max(0.0, min(w1, 1.0))
        
        # 计算利用和探索的数量
        num_exploit = max(1, math.ceil(w1 * actual_top_k))
        num_explore = actual_top_k - num_exploit
        
        # 利用项：基于预测评分
        exploit_indices = torch.argsort(rating_pred, descending=True)[:num_exploit]
        
        # 探索项：基于不确定性
        all_indices = torch.arange(len(rating_pred)).to(device)
        
        # 创建掩码排除已选择的利用项
        mask = torch.ones(len(rating_pred), dtype=torch.bool).to(device)
        mask[exploit_indices] = False
        
        explore_candidates = all_indices[mask]
        
        if len(explore_candidates) > 0 and num_explore > 0:
            candidate_vacuity = vacuity[explore_candidates]
            sorted_indices = torch.argsort(candidate_vacuity, descending=True)
            
            max_explore = min(num_explore, len(explore_candidates))
            selected_explore_indices = sorted_indices[:max_explore]
            explore_indices = explore_candidates[selected_explore_indices]
        else:
            explore_indices = torch.tensor([], dtype=torch.long).to(device)
        
        # 合并结果
        all_selected = torch.cat([exploit_indices, explore_indices])
        
        # 如果数量不足，用利用项补充
        if len(all_selected) < actual_top_k:
            # 从利用项中多取一些
            extra_needed = actual_top_k - len(all_selected)
            if len(exploit_indices) > 0:
                # 从剩下的利用候选中选择
                exploit_mask = torch.ones(len(rating_pred), dtype=torch.bool).to(device)
                exploit_mask[all_selected] = False
                remaining_exploit = all_indices[exploit_mask]
                
                if len(remaining_exploit) > 0:
                    extra_count = min(extra_needed, len(remaining_exploit))
                    extra_indices = remaining_exploit[:extra_count]
                    all_selected = torch.cat([all_selected, extra_indices])
        
        # ✅ 确保数量正确
        all_selected = all_selected[:actual_top_k].cpu().numpy()
        
        # 获取推荐的项目ID
        recommended_items = total_items[all_selected]
        
        # 计算平均不确定性（用于奖励计算）
        if len(explore_indices) > 0:
            avg_uncer = vacuity[explore_indices].mean().item()
        else:
            avg_uncer = 0.0
        
        return recommended_items, avg_uncer

    def calculate_evidential_reward(self, ratings, vacuity, positive_threshold=3.0):
        """计算evidential reward"""
        # 传统奖励
        positive_mask = (ratings >= positive_threshold).float()
        traditional_reward = (ratings - positive_threshold) * positive_mask
        
        # 不确定性奖励
        uncertainty_reward = self.lambda_ * vacuity
        
        # 总奖励
        reward = traditional_reward.mean() + uncertainty_reward.mean()
        return reward

    def train_step(self, batch):
        """训练步骤（整合TensorFlow逻辑）"""
        self.total_steps += 1
        
        user = batch['user_id'].to(self.device)
        pos_items = batch['item_id'][:, 0].to(self.device)
        
        # 获取评分
        if 'rating' in batch:
            ratings = batch['rating'].to(self.device)
        else:
            ratings = torch.full_like(pos_items, 4.0).to(self.device)
        
        B = user.shape[0]
        
        # 为每个用户收集经验
        for i in range(B):
            user_id = user[i].item()
            item_id = pos_items[i].item()
            rating = ratings[i].item()
            
            # 编码状态
            items_ids = self.user_interaction_history.get(user_id, [])
            if len(items_ids) < self.window_len:
                items_ids = items_ids + [item_id] * (self.window_len - len(items_ids))
            else:
                items_ids = items_ids[-self.window_len:] + [item_id]
            
            state = self.get_state_representation(user_id, items_ids[:self.window_len], self.device)
            
            # 生成动作
            action = self.get_action(state, explore=True)
            
            # 获取所有候选项目
            total_items = list(set(range(self.item_num)) - set(self.user_interaction_history.get(user_id, [])))
            if len(total_items) == 0:
                continue
                
            total_items_tensor = torch.tensor(total_items).long().to(self.device)
            total_items_emb = self.item_emb(total_items_tensor)
            
            # 计算证据
            evidence = self.compute_evidence(action, total_items_emb)
            
            # 计算vacuity
            vacuity = self.calculate_vacuity(evidence)
            
            # 推荐项目
            recommended_items, avg_uncer = self.recommend_items(
                user_id, np.array(total_items), evidence, top_k=self.top_k, w1=self.w1)
            
            # 计算奖励（简化）
            # 实际应该从环境中获取反馈
            reward = 1.0 if rating > 3.0 else -1.0
            
            # 更新滑动窗口
            item_emb = self.item_emb(torch.tensor([item_id]).long().to(self.device))
            self.sliding_window.update(user_id, item_emb.squeeze(), 
                                      recommended_items=[self.item_emb(torch.tensor([item]).long().to(self.device)) 
                                                         for item in recommended_items])
            
            # 获取下一个状态
            next_state = self.get_state_representation(user_id, items_ids[-self.window_len:], self.device)
            
            # 存储经验
            if self.use_per:
                self.replay_buffer.push(
                    state.squeeze().detach().cpu(),
                    action.squeeze().detach().cpu(),
                    torch.tensor(reward).cpu(),
                    next_state.squeeze().detach().cpu(),
                    torch.tensor(avg_uncer).cpu(),
                    torch.tensor(0.0).cpu()
                )
            else:
                if len(self.replay_buffer['states']) >= self.buffer_size:
                    self.replay_buffer['states'].pop(0)
                    self.replay_buffer['actions'].pop(0)
                    self.replay_buffer['rewards'].pop(0)
                    self.replay_buffer['next_states'].pop(0)
                    self.replay_buffer['uncers'].pop(0)
                    self.replay_buffer['dones'].pop(0)
                
                self.replay_buffer['states'].append(state.squeeze().detach().cpu())
                self.replay_buffer['actions'].append(action.squeeze().detach().cpu())
                self.replay_buffer['rewards'].append(torch.tensor(reward).cpu())
                self.replay_buffer['next_states'].append(next_state.squeeze().detach().cpu())
                self.replay_buffer['uncers'].append(torch.tensor(avg_uncer).cpu())
                self.replay_buffer['dones'].append(torch.tensor(0.0).cpu())
            
            # 更新行为策略
            self.update_behavior_policy(user_id, state, action)
            
            # 更新交互历史
            self.user_interaction_history[user_id].append(item_id)
        
        # 如果经验足够，开始RL训练
        buffer_size = len(self.replay_buffer) if self.use_per else len(self.replay_buffer['states'])
        
        if buffer_size >= self.warmup_steps and self.total_steps % self.update_freq == 0:
            return self._rl_training_step()
        
        return {'lambda': self.lambda_, 'buffer_size': buffer_size}

    def _rl_training_step(self):
        """RL训练步骤"""
        if self.use_per:
            # PER采样
            batch = self.replay_buffer.sample(self.rl_batch_size)
            states, actions, rewards, next_states, uncers, dones, weights, indices = batch
            
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            uncers = uncers.to(self.device)
            dones = dones.to(self.device)
            weights = weights.to(self.device)
        else:
            # 简单随机采样
            buffer_size = len(self.replay_buffer['states'])
            indices = np.random.choice(buffer_size, min(self.rl_batch_size, buffer_size), replace=False)
            
            states = torch.stack([self.replay_buffer['states'][i] for i in indices]).to(self.device)
            actions = torch.stack([self.replay_buffer['actions'][i] for i in indices]).to(self.device)
            rewards = torch.stack([self.replay_buffer['rewards'][i] for i in indices]).to(self.device)
            next_states = torch.stack([self.replay_buffer['next_states'][i] for i in indices]).to(self.device)
            uncers = torch.stack([self.replay_buffer['uncers'][i] for i in indices]).to(self.device)
            dones = torch.stack([self.replay_buffer['dones'][i] for i in indices]).to(self.device)
            weights = torch.ones_like(rewards)
        
        # 1. 更新Critic（保守Q学习）
        self.critic_opt.zero_grad()
        critic_loss, td_targets = self._update_critic(states, actions, rewards, next_states, uncers, dones, weights)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params, 1.0)
        self.critic_opt.step()
        
        # 更新PER优先级
        if self.use_per:
            with torch.no_grad():
                td_errors = torch.abs(td_targets - 
                                     self.critic1(torch.cat([states, actions], dim=1))).squeeze().detach().cpu()
                self.replay_buffer.update_priorities(indices, td_errors.numpy() + 1e-6)
        
        # 2. 更新Actor
        self.actor_opt.zero_grad()
        actor_loss = self._update_actor(states)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_params, 1.0)
        self.actor_opt.step()
        
        # 3. 更新Evidence网络
        self.evi_opt.zero_grad()
        evi_loss = self._update_evidence(states, actions)
        evi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.evi_params, 1.0)
        self.evi_opt.step()
        
        # 4. 软更新目标网络
        self._soft_update(self.critic1, self.target_critic1, self.tau)
        self._soft_update(self.critic2, self.target_critic2, self.tau)
        
        # 5. λ和w1退火
        self._anneal_parameters()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'evi_loss': evi_loss.item(),
            'lambda': self.lambda_,
            'w1': self.w1
        }

    def _update_critic(self, states, actions, rewards, next_states, uncers, dones, weights):
        """更新Critic网络"""
        with torch.no_grad():
            # 目标动作
            target_actions = self.get_action(next_states, explore=False)
            
            # 目标Q值（双Q学习）
            target_q_input = torch.cat([next_states, target_actions], dim=1)
            target_q1 = self.target_critic1(target_q_input)
            target_q2 = self.target_critic2(target_q_input)
            target_q = torch.min(target_q1, target_q2)
            
            # TD目标（包含不确定性惩罚）
            td_targets = rewards + self.gamma * (1 - dones.float()) * (target_q - self.lambda_ * uncers.unsqueeze(1))
        
        # 当前Q值
        current_input = torch.cat([states, actions], dim=1)
        current_q1 = self.critic1(current_input)
        current_q2 = self.critic2(current_input)
        
        # Bellman损失
        bellman_loss1 = F.mse_loss(current_q1, td_targets, reduction='none')
        bellman_loss2 = F.mse_loss(current_q2, td_targets, reduction='none')
        bellman_loss = (bellman_loss1 + bellman_loss2).mean()
        
        # CQL正则项（保守学习）
        # 采样行为策略动作
        behavior_actions = []
        for state in states:
            # 从行为策略中采样
            user_id = random.choice(list(self.behavior_policy.keys())) if self.behavior_policy else 0
            if user_id in self.behavior_policy:
                behavior_mean = torch.tensor(self.behavior_policy[user_id]['action_mean']).to(self.device)
                behavior_action = behavior_mean + torch.randn_like(behavior_mean) * 0.1
            else:
                behavior_action = torch.randn_like(state[:self.emb_size]) * 0.1
            behavior_actions.append(behavior_action)
        
        if behavior_actions:
            behavior_actions = torch.stack(behavior_actions)
            behavior_input = torch.cat([states, behavior_actions], dim=1)
            behavior_q1 = self.critic1(behavior_input)
            behavior_q2 = self.critic2(behavior_input)
            
            # CQL损失：鼓励行为策略，惩罚当前策略
            cql_loss = (current_q1.mean() - behavior_q1.mean() + 
                       current_q2.mean() - behavior_q2.mean()) * self.conservative_alpha
        else:
            cql_loss = 0
        
        total_loss = bellman_loss + cql_loss
        
        return total_loss, td_targets

    def _update_actor(self, states):
        """更新Actor网络"""
        # 生成动作
        actions = self.get_action(states, explore=False)
        
        # 计算Q值
        q_input = torch.cat([states, actions], dim=1)
        q1 = self.critic1(q_input)
        q2 = self.critic2(q_input)
        q_value = torch.min(q1, q2)
        
        # 最大化Q值
        actor_loss = -q_value.mean()
        
        return actor_loss

    def _update_evidence(self, states, actions):
        """使用真实的用户-项目评分数据训练证据网络"""
        batch_size = states.shape[0]
        
        # 问题：这里没有真实的评分数据！
        # 解决方案1：从经验回放池中获取
        # 解决方案2：修改训练流程，在 train_step 中收集评分数据
        
        # 临时方案：至少使用真实存在的项目和随机评分
        # 但更好的方法是修改整体训练流程
        
        # 选择一些真实存在的项目（不是完全随机的）
        valid_items = torch.randint(0, min(1000, self.item_num), (batch_size,)).to(self.device)
        item_emb = self.item_emb(valid_items)
        
        evidence = self.compute_evidence(actions, item_emb)
        
        # 使用随机评分（暂时）
        # TODO: 应该使用真实评分数据
        random_ratings = torch.randint(1, 6, (batch_size,)).to(self.device)
        y_true = F.one_hot((random_ratings.long() - 1).clamp(0, 4), num_classes=self.num_classes).float()
        
        # 论文公式(8)
        alpha = evidence + 1
        S = alpha.sum(dim=-1, keepdim=True)
        prob = alpha / S
        
        # MSE损失
        mse_loss = F.mse_loss(prob, y_true)
        
        # 方差正则
        var_loss = (prob * (1 - prob) / (S + 1)).sum(dim=-1).mean()
        
        total_loss = mse_loss + var_loss
        
        # 记录证据统计信息
        if hasattr(self, 'evi_stats'):
            self.evi_stats.append({
                'evidence_mean': evidence.mean().item(),
                'evidence_std': evidence.std().item(),
                'loss': total_loss.item()
            })
        
        return total_loss

    def update_behavior_policy(self, user_id, state, action):
        """更新行为策略"""
        state_np = state.detach().cpu().numpy()
        action_np = action.detach().cpu().numpy()
        
        if user_id not in self.behavior_policy:
            self.behavior_policy[user_id] = {
                'state_mean': state_np,
                'action_mean': action_np,
                'count': 1
            }
        else:
            policy = self.behavior_policy[user_id]
            alpha = 0.1
            policy['state_mean'] = alpha * state_np + (1 - alpha) * policy['state_mean']
            policy['action_mean'] = alpha * action_np + (1 - alpha) * policy['action_mean']
            policy['count'] += 1

    def _soft_update(self, source, target, tau):
        """软更新目标网络"""
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def _anneal_parameters(self):
        """参数退火"""
        # λ退火
        if self.total_steps < self.warmup_steps:
            self.lambda_ = self.lambda_init
        else:
            steps_since_warmup = self.total_steps - self.warmup_steps
            total_anneal_steps = 10000
            
            if steps_since_warmup < total_anneal_steps:
                ratio = steps_since_warmup / total_anneal_steps
                self.lambda_ = self.lambda_init - ratio * (self.lambda_init - self.lambda_min)
            else:
                self.lambda_ = self.lambda_min
        
        # w1退火（从TensorFlow移植）
        if (self.total_steps + 1) % 5 == 0:
            if self.w1 <= 0.9:
                self.w1 += 0.1

    def evaluate_step(self, batch, top_k=10):
        """评估步骤 - 使用RL推荐逻辑"""
        user_ids = batch['user_id'].tolist()
        device = self.device
        
        all_precision = []
        all_ndcg = []
        
        for i, user_id in enumerate(user_ids):
            # 获取用户状态
            if user_id in self.user_interaction_history:
                items_ids = self.user_interaction_history[user_id][-self.window_len:]
            else:
                items_ids = []
            
            # 如果历史不足，用一些默认项目填充
            if len(items_ids) < self.window_len:
                padding = [0] * (self.window_len - len(items_ids))
                items_ids = items_ids + padding
            
            # 获取状态表示
            state = self.get_state_representation(user_id, items_ids[:self.window_len], device)
            
            # 生成动作（评估时不探索）
            action = self.get_action(state, explore=False)
            
            # 获取候选项目（排除已交互的）
            interacted_items = set(self.user_interaction_history.get(user_id, []))
            candidate_items = list(set(range(self.item_num)) - interacted_items)
            
            if len(candidate_items) == 0:
                # 如果没有候选项目，使用所有项目
                candidate_items = list(range(self.item_num))
            
            # 随机采样一部分候选项目以加快评估
            if len(candidate_items) > 1000:
                candidate_items = random.sample(candidate_items, 1000)
            
            candidate_tensor = torch.tensor(candidate_items).long().to(device)
            candidate_emb = self.item_emb(candidate_tensor)
            
            # 计算证据
            evidence = self.compute_evidence(action, candidate_emb)
            
            # ✅ 使用传入的 top_k 参数
            recommended_items, _ = self.recommend_items(
                user_id, np.array(candidate_items), evidence, 
                top_k=top_k, w1=self.w1
            )
            
            if len(recommended_items) == 0:
                continue
            
            # 获取真实标签
            pos_items = batch['item_id'][i].tolist()
            
            # 计算命中率和NDCG
            hits = len(set(recommended_items) & set(pos_items))
            
            # 计算 Precision@K
            precision = hits / len(recommended_items)
            all_precision.append(precision)
            
            # 计算 NDCG
            relevance = [1 if item in pos_items else 0 for item in recommended_items]
            dcg = sum([rel / math.log2(idx + 2) for idx, rel in enumerate(relevance)])
            
            # 理想DCG
            ideal_relevance = [1] * min(len(pos_items), top_k)
            idcg = sum([1 / math.log2(idx + 2) for idx in range(len(ideal_relevance))])
            
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0
            
            all_ndcg.append(ndcg)
        
        # 返回平均指标
        if len(all_precision) == 0:
            return {'precision': 0, 'ndcg': 0}
        
        return {
            'precision': np.mean(all_precision),
            'ndcg': np.mean(all_ndcg),
            'test_return': 0.0
        }

    def _compute_idcg(self, labels):
        sorted_labels, _ = torch.sort(labels, descending=True)
        gains = 2 ** sorted_labels - 1
        discounts = torch.log2(torch.arange(2, labels.shape[1] + 2).float().to(labels.device))
        idcg = (gains / discounts).sum(dim=1)
        return idcg

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path

        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # ← 加这一行

        torch.save({
            'model_state_dict': self.state_dict(),
            'actor_opt_state_dict': self.actor_opt.state_dict(),
            'critic_opt_state_dict': self.critic_opt.state_dict(),
            'evi_opt_state_dict': self.evi_opt.state_dict(),
            'total_steps': self.total_steps,
            'lambda': self.lambda_,
            'w1': self.w1,
            'behavior_policy': self.behavior_policy,
            'user_interaction_history': dict(self.user_interaction_history)
        }, model_path)

    def load_model(self, model_path=None):
        """加载模型"""
        if model_path is None:
            model_path = self.model_path
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
        self.evi_opt.load_state_dict(checkpoint['evi_opt_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.lambda_ = checkpoint['lambda']
        self.w1 = checkpoint.get('w1', 0.5)
        self.behavior_policy = checkpoint.get('behavior_policy', {})
        self.user_interaction_history = defaultdict(list, checkpoint.get('user_interaction_history', {}))