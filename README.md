1，	下载文件到本地
2，	在conda命令行中切换到主文件夹下运行conda env create -f environment.yml
3，	运行data/MovieLens_1M文件夹下的MovieLens-1M.ipynb，生成MovieLens_1M数据集
4，	在命令行运行python src/main.py --model_name DRRFull_ECQL_v2 --dataset Grocery_and_Gourmet_Food。即可运行成功
5，	在运行命令中可以指定不同的数据集如:--dataset MovieLens_1M可以用MovieLens_1M数据集进行训练和测试。
6，	在运行命令后可以设置不同的超参数，具体可用的超参数如下：
 
1. 网络结构参数
emb_size: 嵌入维度 = 100

hidden_dim: 隐藏层维度 = 128

window_len: 滑动窗口长度 = 10

2. 训练参数
actor_lr: Actor网络学习率 = 0.001

critic_lr: Critic网络学习率 = 0.001

evi_lr: 证据网络学习率 = 0.001

rl_batch_size: RL训练批大小 = 1

buffer_size: 经验回放缓冲区大小 = 10000

warmup_steps: 预热步数 = 1000

update_freq: 更新频率 = 4

3. 强化学习参数
tau: 目标网络软更新系数 = 0.001

discount_factor: 折扣因子 γ = 0.9

lambda_init: λ初始值 = 1.0

lambda_min: λ最小值 = 0.1

top_k: 推荐项目数 = 10

4. 正则化参数
alpha: α参数 = 0.2

conservative_alpha: 保守学习α = 5.0

5. 模块开关参数
use_rnn: 是否使用RNN = 1 (是)

use_per: 是否使用PER = 1 (是)

use_evidence: 是否使用证据网络 = 1 (是)

use_cql: 是否使用保守Q学习 = 1 (是)

use_windowslen: 是否将推荐加入滑动窗口 = 1 (是)