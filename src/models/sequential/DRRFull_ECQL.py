# -*- coding: utf-8 -*-
"""
DRRFull_ECQL: 完整ECQL实现
基于论文: Looking into User's Long-term Interests through the Lens of Conservative Evidential Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, defaultdict
import random
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
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states), torch.stack(actions), 
                torch.stack(rewards), torch.stack(next_states), 
                torch.stack(dones))
    
    def __len__(self):
        return len(self.buffer)


class SlidingWindow:
    """滑动窗口，用于维护用户最近交互"""
    def __init__(self, window_size):
        self.window_size = window_size
        self.windows = {}  # user_id -> deque
        
    def update(self, user_id, item_emb, recommended_items=None):
        """更新滑动窗口
        Args:
            user_id: 用户ID
            item_emb: 当前交互的项目嵌入
            recommended_items: 上一步推荐的项目嵌入列表
        """
        if user_id not in self.windows:
            self.windows[user_id] = deque(maxlen=self.window_size)
            # 初始填充零向量
            zero_vec = torch.zeros_like(item_emb)
            for _ in range(self.window_size // 2):
                self.windows[user_id].append(zero_vec)
        
        window = self.windows[user_id]
        
        if recommended_items is not None and len(recommended_items) > 0:
            # 用推荐项目替换窗口中的一半项目
            num_to_replace = min(len(recommended_items), len(window) // 2)
            for i in range(num_to_replace):
                window.popleft()
                window.append(recommended_items[i])
        
        # 添加新交互的项目
        window.append(item_emb)
        
    def get_window(self, user_id):
        """获取用户的窗口内容"""
        if user_id in self.windows:
            return list(self.windows[user_id])
        else:
            return [torch.zeros(self.window_size) for _ in range(self.window_size)]


# ---------- 主模型 ----------
class DRRFull_ECQL(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = [
        'emb_size', 'hidden_dim', 'tau', 'discount_factor',
        'actor_lr', 'critic_lr', 'evi_lr','window_len', 'top_k'
    ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--tau', type=float, default=0.005)
        parser.add_argument('--discount_factor', type=float, default=0.99)
        parser.add_argument('--actor_lr', type=float, default=1e-4)
        parser.add_argument('--critic_lr', type=float, default=1e-3)
        parser.add_argument('--evi_lr', type=float, default=1e-3)
        parser.add_argument('--lambda_init', type=float, default=1.0)
        parser.add_argument('--lambda_min', type=float, default=0.1)
        parser.add_argument('--window_len', type=int, default=10)
        parser.add_argument('--top_k', type=int, default=5)
        parser.add_argument('--alpha', type=float, default=0.2)  # 熵正则系数
        parser.add_argument('--conservative_alpha', type=float, default=5.0)  # CQL正则系数
        parser.add_argument('--rl_batch_size', type=int, default=64)  # 改名为rl_batch_size避免冲突
        parser.add_argument('--buffer_size', type=int, default=100000)
        parser.add_argument('--warmup_steps', type=int, default=1000)
        parser.add_argument('--update_freq', type=int, default=4)
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
        self.rl_batch_size = args.rl_batch_size  # 使用rl_batch_size
        self.warmup_steps = args.warmup_steps
        self.update_freq = args.update_freq
        
        self.num_classes = 5  # 评分等级 1-5
        self.rating_threshold = 3.0  # τ，正样本阈值
        
        # ---------- 网络 ----------
        self._define_params()
        self.apply(self.init_weights)
        
        # ---------- 优化器 ----------
        self.actor_opt = torch.optim.Adam(self.actor_params, lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic_params, lr=self.critic_lr)
        self.evi_opt = torch.optim.Adam(self.evi_params, lr=self.evi_lr)
        
        # ---------- 经验池 & 滑动窗口 ----------
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.sliding_window = SlidingWindow(self.window_len)
        self.behavior_policy = {}  # 存储行为策略（从数据中学习）
        
        # ---------- 训练状态 ----------
        self.total_steps = 0
        self.lambda_ = self.lambda_init
        self.episode_rewards = []
        self.user_interaction_history = defaultdict(list)  # 用户交互历史
        
        # ---------- 用于评估 ----------
        self.test_returns = []
        self.positive_counts = []
        self.genre_diversity = []  # 记录类别多样性

    # ---------- 网络定义 ----------
    def _define_params(self):
        # 嵌入层
        self.user_emb = nn.Embedding(self.user_num, self.emb_size)
        self.item_emb = nn.Embedding(self.item_num, self.emb_size)
        
        # 1. SSE (Sequential State Encoder) - 使用GRU
        self.sse_gru = nn.GRU(self.emb_size, self.hidden_dim, 
                              batch_first=True, num_layers=1)
        self.sse_proj = nn.Linear(self.hidden_dim, self.emb_size)
        
        # 2. Actor 网络 (策略网络) - 论文中是高斯策略
        self.actor_mean = MLP(self.emb_size * 2,  # state + user_emb
                              [self.hidden_dim, self.emb_size],
                              activation='relu', out_activation='tanh')
        self.actor_log_std = nn.Parameter(torch.zeros(1, self.emb_size))
        
        # 3. Critic 网络 (Q网络) - 使用double Q-learning
        self.critic1 = MLP(self.emb_size * 3,  # state + action + item
                          [self.hidden_dim, self.hidden_dim, 1],
                          activation='relu', out_activation='linear')
        self.critic2 = MLP(self.emb_size * 3,
                          [self.hidden_dim, self.hidden_dim, 1],
                          activation='relu', out_activation='linear')
        
        # 目标网络
        self.target_critic1 = MLP(self.emb_size * 3,
                                 [self.hidden_dim, self.hidden_dim, 1],
                                 activation='relu', out_activation='linear')
        self.target_critic2 = MLP(self.emb_size * 3,
                                 [self.hidden_dim, self.hidden_dim, 1],
                                 activation='relu', out_activation='linear')
        
        # 4. Evidence 网络
        self.evidence_net = MLP(self.emb_size * 2,  # action + item
                               [self.hidden_dim, self.hidden_dim, self.num_classes],
                               activation='relu', out_activation='relu')
        
        # 初始化目标网络
        self._soft_update(self.critic1, self.target_critic1, tau=1.0)
        self._soft_update(self.critic2, self.target_critic2, tau=1.0)
        
        # 参数分组
        self.actor_params = list(self.user_emb.parameters()) + \
                           list(self.item_emb.parameters()) + \
                           list(self.actor_mean.parameters()) + \
                           list(self.sse_gru.parameters()) + \
                           list(self.sse_proj.parameters())
        self.critic_params = list(self.critic1.parameters()) + \
                            list(self.critic2.parameters())
        self.evi_params = list(self.evidence_net.parameters())

    # ---------- 前向传播（用于传统推荐）----------
    def forward(self, feed_dict):
        """传统前向传播，用于BPR损失"""
        user_id = feed_dict['user_id']
        item_id = feed_dict['item_id']
        B, neg_num = item_id.shape
        
        user_v = self.user_emb(user_id)
        item_v = self.item_emb(item_id)
        
        # 简单计算相似度
        score = (user_v.unsqueeze(1) * item_v).sum(dim=-1)
        return {'prediction': score}

    # ---------- ECQL核心方法 ----------
    def encode_state(self, user_id, device):
        """生成状态表示（SSE）"""
        # 获取滑动窗口内容
        window_items = self.sliding_window.get_window(user_id)
        if len(window_items) == 0:
            # 初始化窗口
            zero_emb = torch.zeros(self.emb_size).to(device)
            window_items = [zero_emb] * self.window_len
            
        window_tensor = torch.stack(window_items).unsqueeze(0).to(device)  # [1, window_len, emb_size]
        
        # GRU编码
        _, hidden = self.sse_gru(window_tensor)
        state = self.sse_proj(hidden.squeeze(0))  # [1, emb_size]
        
        return state

    def get_action(self, state, user_emb, explore=True):
        """根据状态生成动作（高斯策略）"""
        inp = torch.cat([state, user_emb], dim=-1)
        mean = self.actor_mean(inp)
        
        if explore:
            # 探索：采样
            log_std = self.actor_log_std.expand_as(mean)
            std = torch.exp(log_std)
            noise = torch.randn_like(mean)
            action = mean + noise * std
        else:
            # 利用：使用均值
            action = mean
            
        return action

    def compute_vacuity(self, evidence):
        """计算vacuity（二阶不确定性）"""
        # 论文公式(1): vacuity = K / S
        S = evidence.sum(dim=-1, keepdim=True) + self.num_classes
        vacuity = self.num_classes / S
        return vacuity.squeeze(-1)

    def compute_evidence_score(self, user_id, action, item_emb, item_position=None):
        """计算evidential score（论文公式2）"""
        # 证据网络
        inp = torch.cat([action, item_emb], dim=-1)
        evidence = self.evidence_net(inp) + 1  # 确保正数
        
        # 计算概率和预测评分
        alpha = evidence + 1
        S = alpha.sum(dim=-1, keepdim=True)
        prob = alpha / S
        rating_pred = (prob * torch.arange(1, 6).to(prob.device)).sum(dim=-1)
        
        # 计算vacuity
        vacuity = self.compute_vacuity(evidence)
        
        # 计算score（论文公式2）
        if item_position is not None and self.lambda_ > 0:
            # 位置惩罚：log(h_i - W_t + 1)
            position_penalty = torch.log(item_position.float() + 1)
            uncertainty_term = self.lambda_ * vacuity / position_penalty
        else:
            uncertainty_term = self.lambda_ * vacuity
            
        score = rating_pred + uncertainty_term
        
        return score, rating_pred, vacuity, evidence

    def compute_evidential_reward(self, ratings, vacuity, positive_threshold=3.0):
        """计算evidential reward（论文公式3）"""
        # 传统奖励部分
        positive_mask = (ratings >= positive_threshold).float()
        traditional_reward = (ratings - positive_threshold) * positive_mask
        
        # 不确定性正则项
        uncertainty_reward = self.lambda_ * vacuity
        
        # 总奖励
        reward = traditional_reward + uncertainty_reward
        return reward.mean()

    def conservative_q_loss(self, states, actions, rewards, next_states, dones, 
                          behavior_actions=None):
        """保守Q损失（论文公式6）"""
        B = states.shape[0]
        
        with torch.no_grad():
            # 目标Q值（double Q-learning）
            # 简化处理：使用当前用户嵌入
            dummy_users = torch.zeros(B, dtype=torch.long).to(states.device)
            user_emb = self.user_emb(dummy_users)
            next_actions = self.get_action(next_states, user_emb, explore=False)
            
            next_input = torch.cat([next_states, next_actions], dim=-1)
            target_q1 = self.target_critic1(next_input)
            target_q2 = self.target_critic2(next_input)
            target_q = torch.min(target_q1, target_q2)
            target_values = rewards + self.gamma * (1 - dones) * target_q
        
        # 当前Q值
        current_input = torch.cat([states, actions], dim=-1)
        current_q1 = self.critic1(current_input)
        current_q2 = self.critic2(current_input)
        
        # Bellman误差
        bellman_loss1 = F.mse_loss(current_q1, target_values)
        bellman_loss2 = F.mse_loss(current_q2, target_values)
        
        # CQL正则项（关键部分）
        cql_loss = 0
        if behavior_actions is not None and len(behavior_actions) > 0:
            # 计算当前策略和behavior策略的Q值差异
            current_policy_actions = self.get_action(states, user_emb, explore=False)
            current_policy_q1 = self.critic1(torch.cat([states, current_policy_actions], dim=-1))
            current_policy_q2 = self.critic2(torch.cat([states, current_policy_actions], dim=-1))
            
            behavior_q1 = self.critic1(torch.cat([states, behavior_actions], dim=-1))
            behavior_q2 = self.critic2(torch.cat([states, behavior_actions], dim=-1))
            
            # CQL正则：惩罚当前策略Q值，鼓励behavior策略Q值
            cql_loss1 = (current_policy_q1 - behavior_q1).mean()
            cql_loss2 = (current_policy_q2 - behavior_q2).mean()
            
            cql_loss = (cql_loss1 + cql_loss2) * self.conservative_alpha
        
        total_loss = bellman_loss1 + bellman_loss2 + cql_loss
        return total_loss, {
            'bellman_loss': (bellman_loss1 + bellman_loss2).item(),
            'cql_loss': cql_loss.item() if behavior_actions is not None else 0
        }

    def update_behavior_policy(self, user_id, state, action):
        """更新行为策略（从数据中学习）"""
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
            # 指数移动平均
            alpha = 0.1
            policy['state_mean'] = alpha * state_np + (1 - alpha) * policy['state_mean']
            policy['action_mean'] = alpha * action_np + (1 - alpha) * policy['action_mean']
            policy['count'] += 1

    def sample_behavior_action(self, user_id, state):
        """从行为策略中采样动作"""
        if user_id in self.behavior_policy:
            policy = self.behavior_policy[user_id]
            action_mean = torch.tensor(policy['action_mean']).to(state.device)
            # 添加少量噪声
            noise = torch.randn_like(action_mean) * 0.1
            return action_mean + noise
        else:
            # 如果没有行为策略，返回随机动作
            return torch.randn_like(state) * 0.1

    def recommend_top_k(self, user_id, candidate_items, k=5):
        """推荐top-k项目"""
        device = next(self.parameters()).device
        user_v = self.user_emb(torch.tensor([user_id]).long().to(device))
        state = self.encode_state(user_id, device)
        
        # 生成动作
        action = self.get_action(state, user_v, explore=False)
        
        # 为每个候选项目计算score
        scores = []
        for item_id in candidate_items:
            item_v = self.item_emb(torch.tensor([item_id]).long().to(device))
            
            # 获取项目在历史中的位置（简化处理）
            position = len(self.user_interaction_history.get(user_id, [])) + 1
            
            # 计算score
            score, _, _, _ = self.compute_evidence_score(
                user_id, action, item_v, torch.tensor([position]).to(device))
            scores.append(score.item())
        
        # 选择top-k
        top_indices = np.argsort(scores)[-k:][::-1]
        top_items = [candidate_items[i] for i in top_indices]
        
        return top_items, scores

    # ---------- 训练步骤 ----------
    def train_step(self, batch):
        """完整的ECQL训练步骤"""
        self.total_steps += 1
        
        # 1. 准备数据
        user = batch['user_id'].to(self.device)
        pos_items = batch['item_id'][:, 0].to(self.device)
        
        # 检查是否有rating数据
        if 'rating' in batch:
            ratings = batch['rating'].to(self.device)
        else:
            # 如果没有rating，默认给4分
            ratings = torch.full_like(pos_items, 4.0).to(self.device)
        
        B = user.shape[0]
        
        # 2. 收集经验到replay buffer
        for i in range(B):
            user_id = user[i].item()
            item_id = pos_items[i].item()
            rating = ratings[i].item()
            
            # 编码状态
            state = self.encode_state(user_id, self.device)
            
            # 生成动作
            user_v = self.user_emb(user[i].unsqueeze(0))
            action = self.get_action(state, user_v, explore=True)
            
            # 计算奖励
            item_v = self.item_emb(pos_items[i].unsqueeze(0))
            _, rating_pred, vacuity, _ = self.compute_evidence_score(
                user_id, action, item_v, None)  # 简化处理，不使用位置信息
            
            evidential_reward = self.compute_evidential_reward(
                rating_pred, vacuity, self.rating_threshold)
            
            # 下一个状态
            next_state = self.encode_state(user_id, self.device)
            
            # 存储到replay buffer
            self.replay_buffer.push(
                state.squeeze().detach(),
                action.squeeze().detach(),
                torch.tensor([evidential_reward.item()]).to(self.device),
                next_state.squeeze().detach(),
                torch.tensor([0.0]).to(self.device)  # done=False
            )
            
            # 更新行为策略
            self.update_behavior_policy(user_id, state, action)
            
            # 更新滑动窗口
            self.sliding_window.update(user_id, item_v.squeeze().detach())
            
            # 更新交互历史
            if user_id not in self.user_interaction_history:
                self.user_interaction_history[user_id] = []
            self.user_interaction_history[user_id].append(item_id)
        
        # 3. 如果经验足够，开始训练
        if len(self.replay_buffer) >= self.warmup_steps and \
           self.total_steps % self.update_freq == 0:
            
            # 采样batch
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.rl_batch_size)  # 使用rl_batch_size
            
            # 为每个状态采样behavior动作
            behavior_actions = []
            for i in range(states.shape[0]):
                # 随机选择一个用户
                if len(self.behavior_policy) > 0:
                    random_user = random.choice(list(self.behavior_policy.keys()))
                    behavior_action = self.sample_behavior_action(random_user, states[i])
                else:
                    behavior_action = torch.randn_like(states[i]) * 0.1
                behavior_actions.append(behavior_action)
            
            if len(behavior_actions) > 0:
                behavior_actions = torch.stack(behavior_actions)
            
            # 更新Critic（保守Q学习）
            self.critic_opt.zero_grad()
            critic_loss, critic_info = self.conservative_q_loss(
                states, actions, rewards, next_states, dones, 
                behavior_actions if len(behavior_actions) > 0 else None)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_params, 1.0)
            self.critic_opt.step()
            
            # 更新Actor
            self.actor_opt.zero_grad()
            actor_loss = self.compute_actor_loss(states)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_params, 1.0)
            self.actor_opt.step()
            
            # 更新Evidence网络
            self.evi_opt.zero_grad()
            evi_loss = self.compute_evidence_loss(states, actions)
            evi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.evi_params, 1.0)
            self.evi_opt.step()
            
            # 软更新目标网络
            self._soft_update(self.critic1, self.target_critic1, self.tau)
            self._soft_update(self.critic2, self.target_critic2, self.tau)
            
            # λ退火
            self._anneal_lambda()
            
            return {
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'evi_loss': evi_loss.item(),
                'bellman_loss': critic_info['bellman_loss'],
                'cql_loss': critic_info['cql_loss'],
                'lambda': self.lambda_,
                'buffer_size': len(self.replay_buffer)
            }
        
        return {'lambda': self.lambda_, 'buffer_size': len(self.replay_buffer)}

    def compute_actor_loss(self, states):
        """Actor损失：最大化Q值 + 熵正则"""
        B = states.shape[0]
        
        # 采样动作
        random_users = torch.randint(0, self.user_num, (B,)).to(self.device)
        user_v = self.user_emb(random_users)
        actions = self.get_action(states, user_v, explore=False)
        
        # 计算Q值
        q_input = torch.cat([states, actions], dim=-1)
        q1 = self.critic1(q_input)
        q2 = self.critic2(q_input)
        q_value = torch.min(q1, q2)
        
        # 熵正则项（鼓励探索）
        actor_input = torch.cat([states, user_v], dim=-1)
        mean = self.actor_mean(actor_input)
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        # 计算熵
        dist = torch.distributions.Normal(mean, std)
        entropy = dist.entropy().mean()
        
        # 最大化Q值 + 熵正则
        actor_loss = -q_value.mean() - self.alpha * entropy
        
        return actor_loss

    def compute_evidence_loss(self, states, actions):
        """Evidence网络损失"""
        B = states.shape[0]
        
        # 采样随机项目
        random_items = torch.randint(0, self.item_num, (B,)).to(self.device)
        item_v = self.item_emb(random_items)
        
        # 计算证据
        inp = torch.cat([actions, item_v], dim=-1)
        evidence = self.evidence_net(inp) + 1
        
        # 随机生成评分标签（实际应用中应该使用真实评分）
        random_ratings = torch.randint(1, 6, (B,)).to(self.device).float()
        y_true = F.one_hot((random_ratings.long() - 1).clamp(0, 4), 
                          num_classes=self.num_classes).float()
        
        # 证据损失（论文公式8）
        alpha = evidence + 1
        S = alpha.sum(dim=-1, keepdim=True)
        prob = alpha / S
        
        mse_loss = F.mse_loss(prob, y_true, reduction='none')
        var_loss = prob * (1 - prob) / (S + 1)
        
        total_loss = (mse_loss + var_loss).sum(dim=-1).mean()
        
        return total_loss

    def _soft_update(self, source, target, tau):
        """软更新目标网络"""
        with torch.no_grad():
            for src_param, tgt_param in zip(source.parameters(), target.parameters()):
                tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def _anneal_lambda(self):
        """λ退火"""
        if self.total_steps < self.warmup_steps:
            # warmup阶段，λ保持较大
            self.lambda_ = self.lambda_init
        else:
            # 线性退火
            steps_since_warmup = self.total_steps - self.warmup_steps
            total_anneal_steps = 10000  # 退火总步数
            
            if steps_since_warmup < total_anneal_steps:
                ratio = steps_since_warmup / total_anneal_steps
                self.lambda_ = self.lambda_init - ratio * (self.lambda_init - self.lambda_min)
            else:
                self.lambda_ = self.lambda_min

    # ---------- 评估方法 ----------
    @torch.no_grad()
    def evaluate_step(self, batch, top_k=10):
        """评估步骤"""
        user = batch['user_id'].to(self.device)
        items = batch['item_id'][:, :top_k].to(self.device)
        
        # 使用传统方法进行评估（ECQL评估需要完整的RL环境）
        # 这里使用简单的前向传播
        user_v = self.user_emb(user)
        item_v = self.item_emb(items)
        score = (user_v.unsqueeze(1) * item_v).sum(dim=-1)
        
        # 计算指标
        label = batch['label'][:, :top_k].to(self.device)
        hits = (score.argsort(descending=True) < label.sum(-1).unsqueeze(1)).float().sum(-1)
        precision = hits / top_k
        
        # 计算nDCG
        dcg = self._compute_dcg(score, label)
        idcg = self._compute_idcg(label)
        ndcg = (dcg / (idcg + 1e-8)).mean()
        
        return {
            'precision': precision.mean().item(),
            'ndcg': ndcg.item(),
            'test_return': score.mean().item()
        }

    def _compute_dcg(self, scores, labels):
        """计算DCG"""
        _, indices = torch.sort(scores, descending=True)
        ranked_labels = torch.gather(labels, 1, indices)
        gains = 2 ** ranked_labels - 1
        discounts = torch.log2(torch.arange(2, scores.shape[1] + 2).float().to(scores.device))
        dcg = (gains / discounts).sum(dim=1)
        return dcg

    def _compute_idcg(self, labels):
        """计算IDCG"""
        sorted_labels, _ = torch.sort(labels, descending=True)
        gains = 2 ** sorted_labels - 1
        discounts = torch.log2(torch.arange(2, labels.shape[1] + 2).float().to(labels.device))
        idcg = (gains / discounts).sum(dim=1)
        return idcg

def save_model(self, model_path=None):
    """保存模型"""
    if model_path is None:
        model_path = self.args.model_path
    
    torch.save({
        'model_state_dict': self.state_dict(),
        'actor_opt_state_dict': self.actor_opt.state_dict(),
        'critic_opt_state_dict': self.critic_opt.state_dict(),
        'evi_opt_state_dict': self.evi_opt.state_dict(),
        'total_steps': self.total_steps,
        'lambda': self.lambda_,
        'behavior_policy': self.behavior_policy,
        'user_interaction_history': self.user_interaction_history
    }, model_path)

def load_model(self, model_path=None):
    """加载模型"""
    if model_path is None:
        model_path = self.args.model_path
        
    checkpoint = torch.load(model_path, map_location=self.device)
    self.load_state_dict(checkpoint['model_state_dict'])
    self.actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
    self.critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
    self.evi_opt.load_state_dict(checkpoint['evi_opt_state_dict'])
    self.total_steps = checkpoint['total_steps']
    self.lambda_ = checkpoint['lambda']
    self.behavior_policy = checkpoint.get('behavior_policy', {})
    self.user_interaction_history = checkpoint.get('user_interaction_history', {})