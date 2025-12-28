1，	下载文件到本地
2，	在命令行中切换到主文件夹下运行conda env create -f environment.yml（需安装anaconda）
3，   继续运行conda activate recommender激活刚刚创建的名为recommender的环境
3，	运行data/MovieLens_1M文件夹下的MovieLens-1M.ipynb，生成MovieLens_1M数据集
4，	在命令行运行python src/main.py --model_name DRRFull_ECQL_v2 --dataset Grocery_and_Gourmet_Food。即可运行成功
5，	在运行命令中可以指定不同的数据集如:--dataset MovieLens_1M可以用MovieLens_1M数据集进行训练和测试。
6，	在运行命令时可以通过 --emb_size 100 或--hidden_dim 128等指定用于训练的模型的超参数，可供选择的参数如下：
 
1. 网络结构参数
emb_size: 嵌入维度

hidden_dim: 隐藏层维度

window_len: 滑动窗口长度

2. 训练参数
actor_lr: Actor网络学习率

critic_lr: Critic网络学习率

evi_lr: 证据网络学习率

rl_batch_size: RL训练批大小

buffer_size: 经验回放缓冲区大小

warmup_steps: 预热步数

update_freq: 更新频率

3. 强化学习参数
tau: 目标网络软更新系数 

discount_factor: 折扣因子

lambda_init: λ初始值 

lambda_min: λ最小值 

top_k: 推荐项目数

4. 正则化参数
alpha: α参数

conservative_alpha: 保守学习

5. 模块开关参数
use_rnn: 是否使用RNN 

use_per: 是否使用PER

use_evidence: 是否使用证据网络

use_cql: 是否使用保守Q学习

use_windowslen: 是否将推荐加入滑动窗口