import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast
import json
from datetime import datetime
import os
import re

class RecommendationAnalyzer:
    def __init__(self, train_path, rec_path, test_path):
        """
        初始化分析器
        """
        self.train_path = train_path
        self.rec_path = rec_path
        self.test_path = test_path
        
        print("正在加载数据...")
        self.train_df = self._load_train_data()
        self.rec_df = self._load_rec_data()
        self.test_df = self._load_test_data()
        
        self._clean_and_prepare_data()
        
        print("\n" + "="*60)
        print("数据加载和预处理完成！")
        print("="*60)
        print(f"训练数据: {len(self.train_df):,} 条记录")
        print(f"推荐数据: {len(self.rec_df):,} 条记录")
        print(f"测试数据: {len(self.test_df):,} 条记录")
    
    def _clean_and_prepare_data(self):
        """清理和准备数据"""
        # 清理训练数据
        if not self.train_df.empty:
            self.train_df = self.train_df.dropna(subset=['user_id', 'item_id'])
            self.train_df['user_id'] = pd.to_numeric(self.train_df['user_id'], errors='coerce')
            self.train_df['item_id'] = pd.to_numeric(self.train_df['item_id'], errors='coerce')
            self.train_df = self.train_df.dropna(subset=['user_id', 'item_id'])
            self.train_df['user_id'] = self.train_df['user_id'].astype('int64')
            self.train_df['item_id'] = self.train_df['item_id'].astype('int64')
        
        # 清理测试数据
        if not self.test_df.empty:
            self.test_df = self.test_df.dropna(subset=['user_id', 'item_id'])
            self.test_df['user_id'] = pd.to_numeric(self.test_df['user_id'], errors='coerce')
            self.test_df['item_id'] = pd.to_numeric(self.test_df['item_id'], errors='coerce')
            self.test_df = self.test_df.dropna(subset=['user_id', 'item_id'])
            self.test_df['user_id'] = self.test_df['user_id'].astype('int64')
            self.test_df['item_id'] = self.test_df['item_id'].astype('int64')
    
    def _parse_list_robust(self, s):
        """解析列表字符串"""
        if pd.isna(s) or s is None:
            return []
        
        s = str(s).strip()
        if s == '' or s.lower() in ['nan', 'neg_items']:
            return []
        
        try:
            return ast.literal_eval(s)
        except:
            # 尝试提取数字
            numbers = re.findall(r'-?\d+\.?\d*', s)
            result = []
            for num in numbers:
                try:
                    if '.' in num:
                        result.append(float(num))
                    else:
                        result.append(int(num))
                except:
                    continue
            return result
    
    def _load_train_data(self):
        """加载训练数据"""
        try:
            df = pd.read_csv(self.train_path, sep='\t', header=None, 
                           names=['user_id', 'item_id', 'time'])
            print(f"✓ 训练数据: {len(df):,} 行")
            return df
        except Exception as e:
            print(f"✗ 加载训练数据失败: {e}")
            return pd.DataFrame()
    
    def _load_rec_data(self):
        """加载推荐数据"""
        try:
            df = pd.read_csv(self.rec_path, sep='\t')
            
            # 确保列名正确
            if 'user_id' not in df.columns and len(df.columns) >= 1:
                df = df.rename(columns={df.columns[0]: 'user_id'})
            if 'rec_items' not in df.columns and len(df.columns) >= 2:
                df = df.rename(columns={df.columns[1]: 'rec_items'})
            if 'rec_predictions' not in df.columns and len(df.columns) >= 3:
                df = df.rename(columns={df.columns[2]: 'rec_predictions'})
            
            # 解析列表
            if 'rec_items' in df.columns:
                df['rec_items'] = df['rec_items'].apply(self._parse_list_robust)
            if 'rec_predictions' in df.columns:
                df['rec_predictions'] = df['rec_predictions'].apply(self._parse_list_robust)
            
            print(f"✓ 推荐数据: {len(df):,} 行")
            return df
        except Exception as e:
            print(f"✗ 加载推荐数据失败: {e}")
            return pd.DataFrame()
    
    def _load_test_data(self):
        """加载测试数据"""
        try:
            df = pd.read_csv(self.test_path, sep='\t', header=None,
                           names=['user_id', 'item_id', 'time', 'neg_items'])
            print(f"✓ 测试数据: {len(df):,} 行")
            return df
        except Exception as e:
            print(f"✗ 加载测试数据失败: {e}")
            return pd.DataFrame()
    
    def analyze_user_1_detailed(self):
        """详细分析用户1的历史记录和推荐商品"""
        print("\n" + "="*80)
        print("详细分析: 用户ID = 1")
        print("="*80)
        
        # 获取用户1的历史记录
        user_history = self.train_df[self.train_df['user_id'] == 1]
        
        if len(user_history) == 0:
            print("未找到用户1的历史记录")
            # 查看有哪些用户
            print(f"训练数据中的用户示例: {self.train_df['user_id'].unique()[:10]}")
            return
        
        # 获取用户1的推荐
        user_recommendations = self.rec_df[self.rec_df['user_id'] == 1]
        
        if len(user_recommendations) == 0:
            print("未找到用户1的推荐")
            print(f"推荐数据中的用户示例: {self.rec_df['user_id'].unique()[:10]}")
            return
        
        rec_row = user_recommendations.iloc[0]
        recommended_items = rec_row['rec_items']
        prediction_scores = rec_row.get('rec_predictions', [])
        
        print(f"\n用户1的历史交互记录:")
        print(f"  交互项目数: {len(user_history)}")
        print(f"  历史项目ID: {sorted(user_history['item_id'].tolist())}")
        
        print(f"\n用户1的推荐结果:")
        print(f"  推荐项目数: {len(recommended_items)}")
        print(f"  前20个推荐项目: {recommended_items[:20]}")
        
        if prediction_scores:
            print(f"  前20个预测分数: {prediction_scores[:20]}")
            print(f"  预测分数类型: {type(prediction_scores)}")
            print(f"  预测分数长度: {len(prediction_scores)}")
        
        # 确保预测分数长度与推荐项目一致
        if prediction_scores and len(prediction_scores) != len(recommended_items):
            print(f"\n⚠️ 警告: 预测分数长度({len(prediction_scores)})与推荐项目数({len(recommended_items)})不一致!")
            # 取较短的长度
            min_len = min(len(prediction_scores), len(recommended_items))
            if min_len > 0:
                print(f"  使用前{min_len}个项目进行分析")
                recommended_items = recommended_items[:min_len]
                prediction_scores = prediction_scores[:min_len]
        
        # 分析重叠项目
        historical_items = set(user_history['item_id'])
        recommended_set = set(recommended_items)
        overlap = historical_items.intersection(recommended_set)
        
        print(f"\n历史与推荐重叠分析:")
        print(f"  历史项目数: {len(historical_items)}")
        print(f"  推荐项目数: {len(recommended_set)}")
        print(f"  重叠项目数: {len(overlap)}")
        print(f"  重叠项目: {sorted(list(overlap))}")
        
        if overlap:
            # 查找重叠项目的排名
            print(f"\n重叠项目在推荐列表中的排名:")
            for item in sorted(list(overlap)):
                if item in recommended_items:
                    rank = recommended_items.index(item) + 1
                    if prediction_scores and rank <= len(prediction_scores):
                        score = prediction_scores[rank-1]
                        print(f"  项目 {item}: 排名第{rank}位, 预测分数: {score:.4f}")
                    else:
                        print(f"  项目 {item}: 排名第{rank}位")
        
        # 项目流行度分析
        print(f"\n项目流行度分析:")
        all_item_counts = self.train_df['item_id'].value_counts()
        
        # 历史项目的流行度
        hist_popularity = []
        for item in historical_items:
            hist_popularity.append(all_item_counts.get(item, 0))
        
        # 推荐项目的流行度（前50个）
        rec_popularity = []
        for item in recommended_items[:50]:
            rec_popularity.append(all_item_counts.get(item, 0))
        
        print(f"  历史项目平均流行度: {np.mean(hist_popularity):.2f}")
        print(f"  推荐项目平均流行度: {np.mean(rec_popularity):.2f}")
        print(f"  推荐冷启动项目数: {sum(1 for p in rec_popularity if p == 0)}")
        
        # 创建可视化
        self._create_user_1_visualizations(user_history, recommended_items, prediction_scores, 
                                          hist_popularity, rec_popularity, overlap)
        
        # 分析推荐项目的多样性
        self._analyze_recommendation_diversity(recommended_items, prediction_scores)
    
    def _create_user_1_visualizations(self, user_history, recommended_items, prediction_scores,
                                     hist_popularity, rec_popularity, overlap):
        """为用户1创建可视化图表"""
        
        # 设置中文字体（如果需要）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 历史项目与推荐项目的关系图
        fig = plt.figure(figsize=(20, 15))
        
        # 1.1 历史项目分布
        ax1 = plt.subplot(3, 3, 1)
        if len(user_history) > 0:
            hist_items = user_history['item_id'].value_counts().head(min(10, len(user_history)))
            ax1.bar(range(len(hist_items)), hist_items.values)
            ax1.set_xticks(range(len(hist_items)))
            ax1.set_xticklabels([str(int(x)) for x in hist_items.index], rotation=45)
            ax1.set_xlabel('项目ID')
            ax1.set_ylabel('交互次数')
            ax1.set_title('用户1最常交互的项目')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, '无历史数据', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('用户1历史项目')
        
        # 1.2 推荐分数分布
        ax2 = plt.subplot(3, 3, 2)
        if prediction_scores and len(prediction_scores) > 0:
            ax2.hist(prediction_scores, bins=30, alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='分数=0')
            ax2.set_xlabel('预测分数')
            ax2.set_ylabel('频次')
            ax2.set_title(f'用户1推荐分数分布 ({len(prediction_scores)}个项目)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '无预测分数数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('推荐分数分布')
        
        # 1.3 历史vs推荐流行度对比
        ax3 = plt.subplot(3, 3, 3)
        if hist_popularity and rec_popularity:
            positions = np.arange(2)
            means = [np.mean(hist_popularity), np.mean(rec_popularity)]
            stds = [np.std(hist_popularity), np.std(rec_popularity)]
            
            bars = ax3.bar(positions, means, yerr=stds, capsize=10, alpha=0.7)
            bars[0].set_color('blue')
            bars[1].set_color('green')
            ax3.set_xticks(positions)
            ax3.set_xticklabels(['历史项目', '推荐项目'])
            ax3.set_ylabel('平均流行度（交互次数）')
            ax3.set_title('历史 vs 推荐项目流行度对比')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '无流行度数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('流行度对比')
        
        # 1.4 推荐项目排名与分数关系
        ax4 = plt.subplot(3, 3, 4)
        if prediction_scores and len(prediction_scores) > 0:
            # 确保长度一致
            min_len = min(len(recommended_items), len(prediction_scores))
            if min_len > 0:
                ranks = np.arange(1, min(51, min_len) + 1)
                scores = prediction_scores[:len(ranks)]
                
                if len(ranks) == len(scores):
                    scatter = ax4.scatter(ranks, scores, c=scores, cmap='viridis', alpha=0.6)
                    
                    # 标记重叠项目
                    overlap_in_top = []
                    if overlap:
                        for item in overlap:
                            if item in recommended_items[:len(ranks)]:
                                idx = recommended_items.index(item)
                                if idx < len(ranks):
                                    overlap_in_top.append((idx, item))
                    
                    for idx, item in overlap_in_top[:5]:  # 只标记前5个
                        ax4.scatter(idx+1, prediction_scores[idx], 
                                  color='red', s=100, edgecolor='black', linewidth=2,
                                  label=f'重叠项目{item}' if overlap_in_top.index((idx, item)) == 0 else "")
                    
                    ax4.set_xlabel('推荐排名')
                    ax4.set_ylabel('预测分数')
                    ax4.set_title(f'推荐排名 vs 预测分数（前{len(ranks)}个）')
                    ax4.grid(True, alpha=0.3)
                    if overlap_in_top:
                        ax4.legend()
                    plt.colorbar(scatter, ax=ax4)
                else:
                    ax4.text(0.5, 0.5, f'数据长度不一致\nranks: {len(ranks)}, scores: {len(scores)}', 
                            ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, '无预测分数数据', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('排名 vs 分数')
        
        # 1.5 重叠项目展示
        ax5 = plt.subplot(3, 3, 5)
        if overlap:
            overlap_items = list(overlap)
            overlap_counts = []
            for item in overlap_items:
                if item in recommended_items:
                    idx = recommended_items.index(item)
                    overlap_counts.append(idx + 1)  # 排名（从1开始）
            
            if overlap_counts:
                ax5.bar(range(len(overlap_items)), overlap_counts, alpha=0.7, color='orange')
                ax5.set_xticks(range(len(overlap_items)))
                ax5.set_xticklabels([str(int(x)) for x in overlap_items], rotation=45)
                ax5.set_xlabel('重叠项目ID')
                ax5.set_ylabel('推荐排名')
                ax5.set_title('重叠项目在推荐中的排名')
                ax5.grid(True, alpha=0.3)
                
                # 添加排名数值
                for i, v in enumerate(overlap_counts):
                    ax5.text(i, v + 0.5, str(v), ha='center', va='bottom')
            else:
                ax5.text(0.5, 0.5, '无排名数据', ha='center', va='center', transform=ax5.transAxes)
        else:
            ax5.text(0.5, 0.5, '无重叠项目', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('历史与推荐重叠情况')
        
        # 1.6 推荐项目流行度分布
        ax6 = plt.subplot(3, 3, 6)
        if rec_popularity:
            ax6.hist(rec_popularity, bins=30, alpha=0.7, edgecolor='black')
            ax6.axvline(x=np.mean(rec_popularity), color='red', linestyle='--', 
                       label=f'平均: {np.mean(rec_popularity):.1f}')
            ax6.set_xlabel('流行度（交互次数）')
            ax6.set_ylabel('频次')
            ax6.set_title('推荐项目流行度分布')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, '无流行度数据', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('推荐项目流行度')
        
        # 1.7 时间序列分析（如果有时戳）
        ax7 = plt.subplot(3, 3, 7)
        if 'time' in user_history.columns and len(user_history) > 1:
            user_history_sorted = user_history.sort_values('time')
            times = user_history_sorted['time'].values
            items = user_history_sorted['item_id'].values
            
            # 转换为相对时间
            if len(times) > 0:
                times_rel = (times - times[0])
                if len(times_rel) > 1:
                    ax7.scatter(times_rel, items, alpha=0.6)
                    ax7.set_xlabel('相对时间')
                    ax7.set_ylabel('项目ID')
                    ax7.set_title('用户1交互时间序列')
                    ax7.grid(True, alpha=0.3)
                else:
                    ax7.text(0.5, 0.5, '时间数据不足', ha='center', va='center', transform=ax7.transAxes)
            else:
                ax7.text(0.5, 0.5, '无时间数据', ha='center', va='center', transform=ax7.transAxes)
        else:
            ax7.text(0.5, 0.5, '无时间序列数据', ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('交互时间序列')
        
        # 1.8 推荐项目预测分数分布（箱线图）
        ax8 = plt.subplot(3, 3, 8)
        if prediction_scores and len(prediction_scores) > 10:
            # 将推荐分为5组
            group_size = len(prediction_scores) // 5
            if group_size > 0:
                groups = []
                for i in range(5):
                    start = i * group_size
                    end = (i + 1) * group_size if i < 4 else len(prediction_scores)
                    groups.append(prediction_scores[start:end])
                
                box_data = [group for group in groups if len(group) > 0]
                if box_data:
                    box = ax8.boxplot(box_data, showfliers=False)
                    ax8.set_xlabel('推荐分组（按排名）')
                    ax8.set_ylabel('预测分数')
                    ax8.set_title('推荐分数分布（按排名分组）')
                    ax8.grid(True, alpha=0.3)
                    
                    # 添加均值线
                    means = [np.mean(group) for group in box_data]
                    ax8.plot(range(1, len(means) + 1), means, 'r--', alpha=0.7, label='均值')
                    ax8.legend()
                else:
                    ax8.text(0.5, 0.5, '分组数据不足', ha='center', va='center', transform=ax8.transAxes)
            else:
                ax8.text(0.5, 0.5, '数据太少无法分组', ha='center', va='center', transform=ax8.transAxes)
        else:
            ax8.text(0.5, 0.5, '无足够预测分数数据', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('分数分布箱线图')
        
        # 1.9 项目交互次数分布
        ax9 = plt.subplot(3, 3, 9)
        if len(self.train_df) > 0:
            item_interactions = self.train_df['item_id'].value_counts()
            if len(item_interactions) > 0:
                ax9.hist(item_interactions.values, bins=50, alpha=0.7, edgecolor='black')
                if hist_popularity:
                    ax9.axvline(x=np.mean(hist_popularity), color='blue', linestyle='--', 
                               label=f'用户1历史平均: {np.mean(hist_popularity):.1f}')
                if rec_popularity:
                    ax9.axvline(x=np.mean(rec_popularity), color='green', linestyle='--', 
                               label=f'用户1推荐平均: {np.mean(rec_popularity):.1f}')
                ax9.set_xlabel('项目交互次数')
                ax9.set_ylabel('频次')
                ax9.set_title('所有项目流行度分布')
                if hist_popularity or rec_popularity:
                    ax9.legend()
                ax9.grid(True, alpha=0.3)
            else:
                ax9.text(0.5, 0.5, '无交互数据', ha='center', va='center', transform=ax9.transAxes)
        else:
            ax9.text(0.5, 0.5, '无训练数据', ha='center', va='center', transform=ax9.transAxes)
        
        plt.suptitle('用户ID=1 详细分析', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('user_1_detailed_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 2. 创建额外的简单图表
        self._create_additional_charts(user_history, recommended_items, prediction_scores, overlap)
    
    def _create_additional_charts(self, user_history, recommended_items, prediction_scores, overlap):
        """创建额外的简单图表"""
        
        # 1. 重叠项目排名图
        if overlap:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 左图：重叠项目排名
            ax1 = axes[0]
            overlap_ranks = []
            overlap_items_sorted = []
            
            for item in sorted(list(overlap)):
                if item in recommended_items:
                    rank = recommended_items.index(item) + 1
                    overlap_ranks.append(rank)
                    overlap_items_sorted.append(item)
            
            if overlap_ranks:
                bars = ax1.bar(range(len(overlap_ranks)), overlap_ranks, alpha=0.7, color='coral')
                ax1.set_xticks(range(len(overlap_ranks)))
                ax1.set_xticklabels([str(int(x)) for x in overlap_items_sorted], rotation=45)
                ax1.set_xlabel('项目ID')
                ax1.set_ylabel('推荐排名')
                ax1.set_title('重叠项目在推荐列表中的排名')
                ax1.grid(True, alpha=0.3)
                
                # 添加数值
                for i, v in enumerate(overlap_ranks):
                    ax1.text(i, v + 0.5, str(v), ha='center', va='bottom')
            
            # 右图：推荐分数趋势
            ax2 = axes[1]
            if prediction_scores and len(prediction_scores) > 0:
                # 取前100个或全部
                n_points = min(100, len(prediction_scores))
                ranks = np.arange(1, n_points + 1)
                scores = prediction_scores[:n_points]
                
                ax2.plot(ranks, scores, 'b-', alpha=0.7, linewidth=2)
                ax2.fill_between(ranks, scores, alpha=0.3, color='blue')
                
                # 标记重叠项目
                if overlap:
                    for item in overlap:
                        if item in recommended_items[:n_points]:
                            idx = recommended_items.index(item)
                            if idx < n_points:
                                ax2.plot(idx+1, prediction_scores[idx], 'ro', markersize=8)
                                ax2.annotate(f'项目{item}', xy=(idx+1, prediction_scores[idx]),
                                           xytext=(10, 10), textcoords='offset points')
                
                ax2.set_xlabel('推荐排名')
                ax2.set_ylabel('预测分数')
                ax2.set_title('推荐分数随排名变化趋势')
                ax2.grid(True, alpha=0.3)
            
            plt.suptitle('用户1推荐分析补充图表', fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig('user_1_additional_charts.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    def _analyze_recommendation_diversity(self, recommended_items, prediction_scores):
        """分析推荐多样性"""
        print(f"\n推荐多样性分析:")
        
        if not recommended_items:
            print("  无推荐项目")
            return
        
        print(f"  推荐项目总数: {len(recommended_items)}")
        print(f"  唯一项目数: {len(set(recommended_items))}")
        print(f"  重复项目比例: {1 - len(set(recommended_items))/len(recommended_items):.2%}")
        
        if prediction_scores:
            print(f"  预测分数范围: [{min(prediction_scores):.4f}, {max(prediction_scores):.4f}]")
            print(f"  平均分数: {np.mean(prediction_scores):.4f}")
            
            # 分数分布
            positive = sum(1 for s in prediction_scores if s > 0)
            negative = sum(1 for s in prediction_scores if s < 0)
            zero = sum(1 for s in prediction_scores if s == 0)
            
            print(f"  正分数项目: {positive} ({positive/len(prediction_scores):.2%})")
            print(f"  负分数项目: {negative} ({negative/len(prediction_scores):.2%})")
            print(f"  零分数项目: {zero} ({zero/len(prediction_scores):.2%})")
    
    def evaluate_overall_performance(self):
        """评估整体性能"""
        print("\n" + "="*60)
        print("整体性能评估")
        print("="*60)
        
        # 查找共同用户
        if self.test_df.empty or self.rec_df.empty:
            print("测试数据或推荐数据为空")
            return {}
        
        test_users = set(self.test_df['user_id'].unique())
        rec_users = set(self.rec_df['user_id'].unique())
        common_users = test_users.intersection(rec_users)
        
        print(f"测试用户数: {len(test_users):,}")
        print(f"推荐用户数: {len(rec_users):,}")
        print(f"共同用户数: {len(common_users):,}")
        
        if len(common_users) == 0:
            print("没有共同用户，无法评估")
            return {}
        
        # 采样部分用户以提高速度
        sample_users = list(common_users)[:min(1000, len(common_users))]
        print(f"使用 {len(sample_users)} 个用户进行评估...")
        
        results = {}
        top_k_values = [1, 5, 10, 20]
        
        for top_k in top_k_values:
            hr_list, prec_list, recall_list, ndcg_list = [], [], [], []
            
            for user_id in sample_users:
                # 获取测试数据
                test_data = self.test_df[self.test_df['user_id'] == user_id]
                if len(test_data) == 0:
                    continue
                
                pos_item = test_data.iloc[0]['item_id']
                
                # 获取推荐
                rec_data = self.rec_df[self.rec_df['user_id'] == user_id]
                if len(rec_data) == 0:
                    continue
                
                rec_row = rec_data.iloc[0]
                rec_items = rec_row['rec_items']
                if not isinstance(rec_items, list) or len(rec_items) == 0:
                    continue
                
                # 检查是否命中
                hit = 1 if pos_item in rec_items[:top_k] else 0
                hr_list.append(hit)
                prec_list.append(1/top_k if hit else 0)
                recall_list.append(1 if hit else 0)
                
                if hit:
                    rank = rec_items.index(pos_item) + 1
                    dcg = 1 / np.log2(rank + 1)
                    ndcg_list.append(dcg / (1 / np.log2(2)))
                else:
                    ndcg_list.append(0)
            
            if hr_list:
                results[top_k] = {
                    'HR': np.mean(hr_list),
                    'Precision': np.mean(prec_list),
                    'Recall': np.mean(recall_list),
                    'NDCG': np.mean(ndcg_list),
                    '用户数': len(hr_list)
                }
        
        # 打印结果
        if results:
            print("\n整体性能指标:")
            print("-"*40)
            for top_k, metrics in results.items():
                print(f"\nTop-K = {top_k} (基于 {metrics['用户数']} 个用户):")
                print(f"  Hit Rate@{top_k}:    {metrics['HR']:.4f}")
                print(f"  Precision@{top_k}:   {metrics['Precision']:.4f}")
                print(f"  Recall@{top_k}:      {metrics['Recall']:.4f}")
                print(f"  NDCG@{top_k}:        {metrics['NDCG']:.4f}")
        else:
            print("没有计算到有效的性能指标")
        
        # 创建性能对比图
        if results:
            self._create_performance_chart(results)
        
        return results
    
    def _create_performance_chart(self, results):
        """创建性能对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        top_k_list = sorted(results.keys())
        
        # 1. Hit Rate
        ax1 = axes[0, 0]
        hr_values = [results[k]['HR'] for k in top_k_list]
        ax1.bar(range(len(top_k_list)), hr_values, alpha=0.7, color='skyblue')
        ax1.set_xticks(range(len(top_k_list)))
        ax1.set_xticklabels([f'Top-{k}' for k in top_k_list])
        ax1.set_ylabel('Hit Rate')
        ax1.set_title('不同Top-K下的Hit Rate')
        ax1.grid(True, alpha=0.3)
        
        for i, v in enumerate(hr_values):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. Precision
        ax2 = axes[0, 1]
        prec_values = [results[k]['Precision'] for k in top_k_list]
        ax2.bar(range(len(top_k_list)), prec_values, alpha=0.7, color='lightgreen')
        ax2.set_xticks(range(len(top_k_list)))
        ax2.set_xticklabels([f'Top-{k}' for k in top_k_list])
        ax2.set_ylabel('Precision')
        ax2.set_title('不同Top-K下的Precision')
        ax2.grid(True, alpha=0.3)
        
        for i, v in enumerate(prec_values):
            ax2.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. Recall
        ax3 = axes[1, 0]
        recall_values = [results[k]['Recall'] for k in top_k_list]
        ax3.bar(range(len(top_k_list)), recall_values, alpha=0.7, color='salmon')
        ax3.set_xticks(range(len(top_k_list)))
        ax3.set_xticklabels([f'Top-{k}' for k in top_k_list])
        ax3.set_ylabel('Recall')
        ax3.set_title('不同Top-K下的Recall')
        ax3.grid(True, alpha=0.3)
        
        for i, v in enumerate(recall_values):
            ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 4. NDCG
        ax4 = axes[1, 1]
        ndcg_values = [results[k]['NDCG'] for k in top_k_list]
        ax4.bar(range(len(top_k_list)), ndcg_values, alpha=0.7, color='gold')
        ax4.set_xticks(range(len(top_k_list)))
        ax4.set_xticklabels([f'Top-{k}' for k in top_k_list])
        ax4.set_ylabel('NDCG')
        ax4.set_title('不同Top-K下的NDCG')
        ax4.grid(True, alpha=0.3)
        
        for i, v in enumerate(ndcg_values):
            ax4.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
        
        plt.suptitle('DRRFull_ECQL_v2 推荐系统性能评估', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('overall_performance.png', dpi=150, bbox_inches='tight')
        plt.show()

# 主程序
def main():
    # 设置文件路径
    train_file = "train.csv"
    rec_file = "rec-DRRFull_ECQL_v2-test.csv"
    test_file = "test.csv"
    
    print("="*80)
    print("DRRFull_ECQL_v2 推荐系统深度分析")
    print("="*80)
    
    # 检查文件是否存在
    for file_path in [train_file, rec_file, test_file]:
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在!")
            return
    
    # 创建分析器
    analyzer = RecommendationAnalyzer(train_file, rec_file, test_file)
    
    # 1. 详细分析用户1
    print("\n" + "="*80)
    print("开始分析用户1...")
    print("="*80)
    
    analyzer.analyze_user_1_detailed()
    
    # 2. 评估整体性能
    print("\n" + "="*80)
    print("开始整体性能评估...")
    print("="*80)
    
    results = analyzer.evaluate_overall_performance()
    
    # 3. 保存分析报告
    print("\n" + "="*80)
    print("生成分析报告...")
    print("="*80)
    
    with open('recommendation_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DRRFull_ECQL_v2 推荐系统分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. 数据统计\n")
        f.write("-"*40 + "\n")
        f.write(f"训练数据: {len(analyzer.train_df):,} 条记录\n")
        f.write(f"推荐数据: {len(analyzer.rec_df):,} 条记录\n")
        f.write(f"测试数据: {len(analyzer.test_df):,} 条记录\n\n")
        
        f.write("2. 用户1分析摘要\n")
        f.write("-"*40 + "\n")
        # 这里可以添加用户1的分析摘要
        
        f.write("\n3. 整体性能评估\n")
        f.write("-"*40 + "\n")
        
        if results:
            for top_k, metrics in results.items():
                f.write(f"\nTop-K = {top_k}:\n")
                f.write(f"  Hit Rate@{top_k}:    {metrics['HR']:.4f}\n")
                f.write(f"  Precision@{top_k}:   {metrics['Precision']:.4f}\n")
                f.write(f"  Recall@{top_k}:      {metrics['Recall']:.4f}\n")
                f.write(f"  NDCG@{top_k}:        {metrics['NDCG']:.4f}\n")
                f.write(f"  评估用户数:         {metrics['用户数']}\n")
        else:
            f.write("无有效的性能评估数据\n")
        
        f.write("\n4. 生成的可视化文件\n")
        f.write("-"*40 + "\n")
        f.write("1. user_1_detailed_analysis.png - 用户1详细分析图\n")
        f.write("2. user_1_additional_charts.png - 用户1补充分析图\n")
        f.write("3. overall_performance.png - 整体性能评估图\n")
        
        f.write("\n5. 分析结论\n")
        f.write("-"*40 + "\n")
        f.write("分析完成！详细结果请查看生成的图表文件。\n")
    
    print("分析报告已保存到: recommendation_analysis_report.txt")
    print("\n生成的可视化文件:")
    print("  1. user_1_detailed_analysis.png - 用户1详细分析图")
    print("  2. user_1_additional_charts.png - 用户1补充分析图")
    print("  3. overall_performance.png - 整体性能评估图")
    print("\n分析完成！")

if __name__ == "__main__":
    main()