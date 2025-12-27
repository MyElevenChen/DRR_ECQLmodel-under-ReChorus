
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

# 设置全局字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class UserRecommendationAnalyzer:
    def __init__(self, user_id, historical_items, model1_data, model2_data):
        """
        初始化用户推荐分析器
        
        Args:
            user_id: 用户ID
            historical_items: 用户历史交互商品集合
            model1_data: 模型1推荐数据 (rec_items, rec_predictions)
            model2_data: 模型2推荐数据 (rec_items, rec_predictions)
        """
        self.user_id = user_id
        self.historical_items = set(historical_items)
        
        # 解析模型1数据
        self.model1_items = model1_data[0]
        self.model1_scores = model1_data[1]
        
        # 解析模型2数据
        self.model2_items = model2_data[0]
        self.model2_scores = model2_data[1]
        
        # 验证数据长度
        if len(self.model1_items) != len(self.model1_scores):
            print(f"警告: 模型1推荐项目数({len(self.model1_items)})与分数数({len(self.model1_scores)})不一致")
            min_len = min(len(self.model1_items), len(self.model1_scores))
            self.model1_items = self.model1_items[:min_len]
            self.model1_scores = self.model1_scores[:min_len]
            
        if len(self.model2_items) != len(self.model2_scores):
            print(f"警告: 模型2推荐项目数({len(self.model2_items)})与分数数({len(self.model2_scores)})不一致")
            min_len = min(len(self.model2_items), len(self.model2_scores))
            self.model2_items = self.model2_items[:min_len]
            self.model2_scores = self.model2_scores[:min_len]
    
    def calculate_hit_rate(self, recommended_items, top_k=None):
        """计算命中率"""
        if top_k:
            items_to_check = recommended_items[:top_k]
        else:
            items_to_check = recommended_items
        
        hit_items = [item for item in items_to_check if item in self.historical_items]
        hit_count = len(hit_items)
        
        # 计算命中率（基于历史项目总数）
        hit_rate = hit_count / len(self.historical_items) if self.historical_items else 0
        
        return {
            'hit_count': hit_count,
            'hit_items': hit_items,
            'hit_rate': hit_rate,
            'checked_count': len(items_to_check)
        }
    
    def analyze_model_performance(self, model_name, recommended_items, prediction_scores):
        """分析单个模型性能"""
        print(f"\n{'-'*60}")
        print(f"{model_name} 性能分析")
        print(f"{'-'*60}")
        
        # 基本统计
        print(f"推荐项目数: {len(recommended_items)}")
        print(f"历史项目数: {len(self.historical_items)}")
        
        # 不同Top-K的命中率
        hit_results = {}
        for top_k in [1, 5, 10, 20, 50, 100]:
            result = self.calculate_hit_rate(recommended_items, top_k)
            hit_results[top_k] = result
            
            print(f"Top-{top_k:3d}: {result['hit_count']:2d}个命中 ({result['hit_rate']:.2%})", end="")
            if result['hit_items']:
                print(f" - 命中项目: {result['hit_items']}")
            else:
                print()
        
        # 命中项目详情
        all_hits = self.calculate_hit_rate(recommended_items)  # 不限制top_k
        if all_hits['hit_items']:
            print(f"\n全部命中项目详情:")
            for item in all_hits['hit_items']:
                rank = recommended_items.index(item) + 1
                score = prediction_scores[rank-1] if rank <= len(prediction_scores) else None
                print(f"  项目 {item}: 排名第{rank}位", end="")
                if score is not None:
                    print(f", 预测分数: {score:.4f}")
                else:
                    print()
        
        # 预测分数统计
        if prediction_scores:
            scores_array = np.array(prediction_scores)
            print(f"\n预测分数统计:")
            print(f"  最高分: {scores_array.max():.4f}")
            print(f"  最低分: {scores_array.min():.4f}")
            print(f"  平均分: {scores_array.mean():.4f}")
            print(f"  中位数: {np.median(scores_array):.4f}")
            print(f"  标准差: {scores_array.std():.4f}")
            
            # 分数分布
            positive = np.sum(scores_array > 0)
            negative = np.sum(scores_array < 0)
            zero = np.sum(scores_array == 0)
            
            print(f"  正分数: {positive} ({positive/len(scores_array):.2%})")
            print(f"  负分数: {negative} ({negative/len(scores_array):.2%})")
            print(f"  零分数: {zero} ({zero/len(scores_array):.2%})")
        
        return hit_results
    
    def compare_models(self):
        """对比两个模型"""
        print(f"\n{'='*80}")
        print(f"用户 {self.user_id} - 两个模型对比分析")
        print(f"{'='*80}")
        
        # 分析每个模型
        print(f"\n历史交互项目: {sorted(self.historical_items)}")
        
        model1_results = self.analyze_model_performance(
            "模型1 (原模型)", 
            self.model1_items, 
            self.model1_scores
        )
        
        model2_results = self.analyze_model_performance(
            "模型2 (新模型)", 
            self.model2_items, 
            self.model2_scores
        )
        
        # 对比分析
        print(f"\n{'='*80}")
        print("模型对比总结")
        print(f"{'='*80}")
        
        comparison_data = []
        for top_k in [1, 5, 10, 20, 50, 100]:
            m1_hits = model1_results[top_k]['hit_count']
            m2_hits = model2_results[top_k]['hit_count']
            m1_rate = model1_results[top_k]['hit_rate']
            m2_rate = model2_results[top_k]['hit_rate']
            
            comparison_data.append({
                'Top-K': top_k,
                'Model1_Hits': m1_hits,
                'Model2_Hits': m2_hits,
                'Model1_HitRate': m1_rate,
                'Model2_HitRate': m2_rate,
                'Difference': m2_hits - m1_hits,
                'Rate_Difference': m2_rate - m1_rate
            })
            
            # 判断哪个模型更好
            if m2_hits > m1_hits:
                better = "模型2更好"
            elif m1_hits > m2_hits:
                better = "模型1更好"
            else:
                better = "两者相同"
            
            print(f"Top-{top_k:3d}: 模型1 {m1_hits:2d}命中 ({m1_rate:.2%}) | "
                  f"模型2 {m2_hits:2d}命中 ({m2_rate:.2%}) | {better}")
        
        # 创建可视化图表
        self.create_comparison_visualizations(comparison_data, model1_results, model2_results)
        
        return pd.DataFrame(comparison_data)
    
    def create_comparison_visualizations(self, comparison_data, model1_results, model2_results):
        """创建对比可视化图表"""
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 不同Top-K命中率对比（折线图）
        top_k_values = [data['Top-K'] for data in comparison_data]
        m1_rates = [data['Model1_HitRate'] for data in comparison_data]
        m2_rates = [data['Model2_HitRate'] for data in comparison_data]
        
        ax1.plot(top_k_values, m1_rates, 'o-', linewidth=2, markersize=6, label='模型1', color='blue')
        ax1.plot(top_k_values, m2_rates, 's-', linewidth=2, markersize=6, label='模型2', color='red')
        ax1.set_xlabel('Top-K', fontsize=12)
        ax1.set_ylabel('命中率', fontsize=12)
        ax1.set_title('不同Top-K命中率对比', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(top_k_values)
        ax1.tick_params(axis='both', labelsize=10)
        
        # 2. 不同Top-K命中数对比（柱状图）
        x = np.arange(len(top_k_values))
        width = 0.35
        
        m1_hits = [data['Model1_Hits'] for data in comparison_data]
        m2_hits = [data['Model2_Hits'] for data in comparison_data]
        
        bars1 = ax2.bar(x - width/2, m1_hits, width, label='DRRFull_ECQL_v2', alpha=0.8, color='blue')
        bars2 = ax2.bar(x + width/2, m2_hits, width, label='SASRec', alpha=0.8, color='red')
        
        ax2.set_xlabel('Top-K', fontsize=12)
        ax2.set_ylabel('命中数量', fontsize=12)
        ax2.set_title('不同Top-K命中数对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Top-{k}' for k in top_k_values], rotation=45)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='both', labelsize=10)
        
        # 添加数值标签
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        autolabel(bars1)
        autolabel(bars2)
        
        # 调整布局
        plt.suptitle(f'用户 {self.user_id} - 模型对比分析', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'user_{self.user_id}_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 创建详细的文本报告
        self.generate_detailed_report(comparison_data, model1_results, model2_results)
    
    def generate_detailed_report(self, comparison_data, model1_results, model2_results):
        """生成详细对比报告"""
        report = f"""
{'='*80}
用户 {self.user_id} - 模型对比详细报告
{'='*80}

历史交互项目 ({len(self.historical_items)}个):
{sorted(self.historical_items)}

模型1 (原模型) 性能总结:
- 推荐项目数: {len(self.model1_items)}
- 总命中数: {model1_results[100]['hit_count']}
- 总命中率: {model1_results[100]['hit_rate']:.2%}
- 命中项目: {model1_results[100]['hit_items'] if model1_results[100]['hit_items'] else '无'}

模型2 (新模型) 性能总结:
- 推荐项目数: {len(self.model2_items)}
- 总命中数: {model2_results[100]['hit_count']}
- 总命中率: {model2_results[100]['hit_rate']:.2%}
- 命中项目: {model2_results[100]['hit_items'] if model2_results[100]['hit_items'] else '无'}

详细对比分析:
"""
        # 添加详细对比
        for data in comparison_data:
            report += f"Top-{data['Top-K']:3d}: "
            report += f"模型1 {data['Model1_Hits']:2d}命中 ({data['Model1_HitRate']:.2%}) | "
            report += f"模型2 {data['Model2_Hits']:2d}命中 ({data['Model2_HitRate']:.2%}) | "
            
            if data['Difference'] > 0:
                report += f"模型2多{data['Difference']}个命中 (+{data['Rate_Difference']:.2%})\n"
            elif data['Difference'] < 0:
                report += f"模型1多{-data['Difference']}个命中 ({data['Rate_Difference']:+.2%})\n"
            else:
                report += "两者相同\n"
        
        # 添加结论
        total_diff = model2_results[100]['hit_count'] - model1_results[100]['hit_count']
        rate_diff = model2_results[100]['hit_rate'] - model1_results[100]['hit_rate']
        
        report += f"\n结论:\n"
        if total_diff > 0:
            report += f"✓ 模型2整体表现更好，多命中{total_diff}个项目，命中率高{rate_diff:.2%}\n"
            
            # 分析优势区间
            best_top_k = max(comparison_data, key=lambda x: x['Difference'])
            report += f"✓ 在Top-{best_top_k['Top-K']}时优势最明显，多命中{best_top_k['Difference']}个\n"
        elif total_diff < 0:
            report += f"✗ 模型1整体表现更好，多命中{-total_diff}个项目，命中率高{-rate_diff:.2%}\n"
        else:
            report += f"○ 两个模型表现相同\n"
        
        # 推荐策略建议
        report += f"\n推荐策略建议:\n"
        if model2_results[100]['hit_count'] > 0:
            hit_items = model2_results[100]['hit_items']
            ranks = [self.model2_items.index(item) + 1 for item in hit_items]
            report += f"- 模型2命中项目排名: {list(zip(hit_items, ranks))}\n"
            
            if min(ranks) <= 10:
                report += f"- 有项目排名在前10，推荐质量较高\n"
            else:
                report += f"- 所有命中项目排名都在10名之后，可以考虑优化排序\n"
        
        # 保存报告
        with open(f'user_{self.user_id}_model_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n详细报告已保存到: user_{self.user_id}_model_comparison_report.txt")
        print(f"可视化图表已保存到: user_{self.user_id}_model_comparison.png")

# 主程序
def main():
    # 用户13134的数据
    user_id = 13134
    historical_items = {5373, 1675, 473, 856, 3039, 3035, 1782, 6759, 6033, 1418}
    
    # 模型1数据（原来的模型）
    model1_items = [2731, 1596, 4345, 3500, 3234, 4704, 4704, 6471, 6943, 1079, 
                    3777, 7153, 6790, 4937, 5316, 6930, 6592, 1469, 7547, 7382, 
                    520, 1073, 183, 2409, 1418, 6066, 6671, 4418, 1778, 4606, 
                    5750, 5837, 5443, 5443, 7877, 1231, 8574, 5598, 8220, 286, 
                    1172, 7696, 3760, 7232, 7232, 389, 7191, 6417, 4914, 2006, 
                    1299, 4349, 4756, 1814, 1483, 7015, 7900, 7519, 6579, 1105, 
                    6677, 4706, 6402, 4906, 4295, 655, 4259, 5885, 4396, 4435, 
                    1010, 1214, 5585, 8376, 7117, 7975, 2863, 6785, 5619, 6688, 
                    954, 2147, 179, 3919, 4379, 6053, 1464, 6771, 4990, 7687, 
                    2631, 2773, 50, 5371, 8188, 2974, 6457, 3853, 5010, 5423]
    
    model1_scores = [3.8928092, 3.7176619, 3.5851343, 3.2939384, 3.0172586, 2.8485227, 
                     2.8485227, 2.6673756, 2.4068894, 2.3122447, 2.3086305, 2.002451, 
                     1.9949673, 1.950294, 1.897567, 1.7684474, 1.7330935, 1.5960693, 
                     1.4046391, 1.3889487, 1.3080174, 1.2974377, 1.2825531, 1.2256014, 
                     1.2104514, 1.1345736, 1.0608176, 0.9642291, 0.94516045, 0.9162797, 
                     0.8673235, 0.80664647, 0.74067277, 0.74067277, 0.7262955, 0.6091418, 
                     0.49091232, 0.4528631, 0.42095235, 0.41591072, 0.39131874, 0.38051373, 
                     0.3762777, 0.34708568, 0.34708568, 0.30361855, 0.2627684, 0.21846616, 
                     0.09714639, 0.08444095, 0.07081416, 0.0043859333, -0.103998154, 
                     -0.1612511, -0.18514991, -0.24872863, -0.27626002, -0.33994842, 
                     -0.36417767, -0.41228595, -0.41546, -0.46719602, -0.47835612, 
                     -0.49101177, -0.6568537, -0.6936121, -0.7054029, -0.73764753, 
                     -0.73924756, -0.7566551, -0.7566667, -0.78531104, -0.8728653, 
                     -0.9541667, -1.0153267, -1.1716076, -1.1762733, -1.1939, -1.2774167, 
                     -1.3295426, -1.3347558, -1.3578582, -1.4622997, -1.5655115, -1.6934707, 
                     -1.7529972, -1.9615816, -1.9987609, -2.0007062, -2.307884, -2.5984218, 
                     -2.6643114, -2.6919765, -2.8617089, -3.0474472, -3.067535, -3.424408, 
                     -3.6554227, -3.8117366, -4.821529]
    
    # 模型2数据（新的模型）
    model2_items = [3234, 1483, 4418, 2731, 1418, 2409, 655, 3919, 1079, 1596, 
                    4937, 183, 6943, 4435, 7153, 6671, 5443, 5443, 1469, 1172, 
                    179, 4704, 4704, 389, 2974, 4756, 1299, 5585, 2006, 7191, 
                    3777, 7547, 8574, 4396, 3853, 3500, 5598, 1814, 6930, 7519, 
                    520, 7877, 7382, 1778, 4606, 4349, 4345, 6771, 8220, 6579, 
                    6785, 7232, 7232, 6402, 5316, 7015, 1214, 2147, 6688, 4706, 
                    286, 7117, 5619, 7696, 6457, 6790, 1073, 5010, 8188, 2863, 
                    6592, 6677, 2631, 5750, 4295, 50, 954, 4906, 1105, 5885, 
                    4990, 5371, 6066, 7900, 4914, 5423, 3760, 4259, 1010, 7687, 
                    6471, 1464, 4379, 6417, 6053, 2773, 8376, 1231, 7975, 5837]
    
    model2_scores = [2.5158553, 2.4920325, 2.0990455, 2.0962498, 2.0218089, 1.9678967, 
                     1.9494281, 1.8640611, 1.7545795, 1.6931876, 1.6873093, 1.6087444, 
                     1.2518631, 1.0868069, 0.98908705, 0.823568, 0.7968136, 0.7968136, 
                     0.6329794, 0.6303605, 0.5901626, 0.5347113, 0.5347113, 0.52391994, 
                     0.5187923, 0.5168102, 0.49505997, 0.483539, 0.45188817, 0.40359396, 
                     0.39526245, 0.3204748, 0.19129634, 0.16005707, 0.15604722, 0.111365974, 
                     0.09792632, 0.065979406, 0.034290552, 0.02036357, 0.016563177, -0.0817551, 
                     -0.14708135, -0.1959393, -0.21131018, -0.24571861, -0.29829893, -0.30259097, 
                     -0.38557485, -0.39186308, -0.4068574, -0.42509937, -0.42509937, -0.50869787, 
                     -0.509585, -0.5435786, -0.56450045, -0.6355789, -0.71677625, -0.7877842, 
                     -0.81260276, -0.8299825, -0.859655, -0.9053391, -0.91671044, -0.9560613, 
                     -0.9734959, -0.9961858, -1.0599973, -1.1074255, -1.1608659, -1.1657689, 
                     -1.1795034, -1.2407874, -1.3111813, -1.3253374, -1.401352, -1.5183874, 
                     -1.5473655, -1.547432, -1.599016, -1.6301421, -1.8290007, -1.900215, 
                     -1.9436843, -2.1814175, -2.2132368, -2.2276206, -2.2535162, -2.2675467, 
                     -2.3494544, -2.5739832, -2.8695803, -2.88013, -3.0483031, -3.2183437, 
                     -3.2341392, -3.3878107, -3.4152973, -3.4914289]
    
    # 创建分析器
    analyzer = UserRecommendationAnalyzer(
        user_id=user_id,
        historical_items=historical_items,
        model1_data=(model1_items, model1_scores),
        model2_data=(model2_items, model2_scores)
    )
    
    # 执行对比分析
    comparison_df = analyzer.compare_models()
    
    # 显示对比数据
    print(f"\n对比数据表格:")
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()