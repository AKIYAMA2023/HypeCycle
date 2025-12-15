# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random

# -----------------------------------------------------------------------------
#  Agent definition
# -----------------------------------------------------------------------------

random.seed(42)
np.random.seed(42)

class Turtle:
    """Single agent with an opinion and a knowledge flag."""

    def __init__(self, x=None, y=None, latent_opinion=None):
        self.x = random.random() if x is None else x
        self.y = random.random() if y is None else y
        self.latent_opinion = np.random.normal(0, 0.5) if latent_opinion is None else latent_opinion  # fixed enthusiasm
        self.expressed_opinion = None  # evolving, only set when aware
        self.know = 0  # 0 = ignorant, 1 = knowledgeable
        self.adopted = False  # 新技術を導入したかどうか
        self.is_adopter = False  # 導入者（Technology Fundamentalに基づく期待を持つ）


# -----------------------------------------------------------------------------
#  Opinion‑dynamics model
# -----------------------------------------------------------------------------

class Model:
    """Opinion dynamics with **no ignorant‑ignorant interactions**.

    ▸ One developper (know=1, opinion=1.0) + `agent_count` ignorant agents.
    ▸ Only pairs that include at least ONE knowledgeable agent can interact.
    ▸ In know=1 vs know=0, only the ignorant side updates & gains knowledge.
    ▸ In know=1 vs know=1, both update.
    ▸ know=0 vs know=0 pairs are skipped entirely.
    """

    def __init__(self, agent_count: int = 999, *, learning_rate: float = 0.3, confidence_threshold: float = 0.5, 
                 adoption_probability: float = 0.01):
        self.agent_count = agent_count
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.adoption_probability = adoption_probability  # 新技術導入の確率

        # runtime containers
        self.turtles = []
        self.tick = 0
        self.sum_opinion = []
        self.know_count = []
        self.opinion_distribution = []
        self.ticks = []
        self.aware_positive_count = []  # 新しい指標: Aware かつ opinion > 0.5 の数
        self.adopter_count = []  # 導入者の数を追跡

    # -------------------------------- setup ---------------------------------
    def setup(self):
        self.turtles.clear()
        self.tick = 0
        self.sum_opinion.clear()
        self.know_count.clear()
        self.opinion_distribution.clear()
        self.ticks.clear()
        self.adopter_count.clear()

        for _ in range(self.agent_count):
            self.turtles.append(Turtle())

        # Developer agent: always aware, latent_opinion=1, expressed_opinion=1
        developer = Turtle(latent_opinion=1.0)
        developer.expressed_opinion = 1.0
        developer.know = 1
        self.turtles.append(developer)

        self._record()

    def get_technology_fundamental(self):
        """Technology Fundamentalを計算（採用者の現在の割合に比例）"""
        total_agents = len(self.turtles)
        adopters = [t for t in self.turtles if t.adopted]
        adoption_rate = len(adopters) / total_agents if total_agents > 0 else 0
        return adoption_rate  # 0から1の間の値

    # -------------------------------- step ----------------------------------
    def step(self):
        # 1. Technology Fundamentalを計算
        tech_fundamental = self.get_technology_fundamental()
        
        # 2. Expressed opinionに基づく新技術導入の判定
        aware_agents = [t for t in self.turtles if t.know == 1 and not t.adopted]
        for agent in aware_agents:
            # 表明意見が負またはゼロなら導入確率はゼロ
            if agent.expressed_opinion <= 0:
                adoption_prob = 0.0
            else:
                # 正の意見の場合のみ導入確率を計算（最大で adoption_probability * 10倍）
                adoption_prob = self.adoption_probability * (1 + agent.expressed_opinion * 9)
            
            if random.random() < adoption_prob:
                agent.adopted = True
                agent.is_adopter = True
                pre_adoption_opinion = agent.expressed_opinion
                agent.expressed_opinion = (1 - self.learning_rate * 2) * pre_adoption_opinion + self.learning_rate * 2 * tech_fundamental

        # 3. 意見の相互作用
        knowledgeable_agents = [t for t in self.turtles if t.know == 1]
        random.shuffle(knowledgeable_agents)

        for a in knowledgeable_agents:
            candidates = [p for p in self.turtles if p is not a]
            b = random.choice(candidates)

            if b.know == 0:
                # ignorant partner: similarity check with latent opinion
                if abs(a.expressed_opinion - b.latent_opinion) < self.confidence_threshold:
                    # update: b becomes aware, forms expressed opinion
                    b.expressed_opinion = (1 - self.learning_rate) * b.latent_opinion + self.learning_rate * a.expressed_opinion
                    b.know = 1
                # else: no update
            elif b.know == 1:
                # 導入者は説得されない、一方的に説得する
                if a.is_adopter and not b.is_adopter:
                    # aが導入者、bが非導入者：aがbを一方的に説得
                    if abs(a.expressed_opinion - b.expressed_opinion) < self.confidence_threshold:
                        b.expressed_opinion = (1 - self.learning_rate) * b.expressed_opinion + self.learning_rate * a.expressed_opinion
                elif b.is_adopter and not a.is_adopter:
                    # bが導入者、aが非導入者：bがaを一方的に説得
                    if abs(a.expressed_opinion - b.expressed_opinion) < self.confidence_threshold:
                        a.expressed_opinion = (1 - self.learning_rate) * a.expressed_opinion + self.learning_rate * b.expressed_opinion
                elif not a.is_adopter and not b.is_adopter:
                    # 両方とも非導入者：通常の相互説得
                    if abs(a.expressed_opinion - b.expressed_opinion) < self.confidence_threshold:
                        new_a = (1 - self.learning_rate) * a.expressed_opinion + self.learning_rate * b.expressed_opinion
                        new_b = (1 - self.learning_rate) * b.expressed_opinion + self.learning_rate * a.expressed_opinion
                        a.expressed_opinion, b.expressed_opinion = new_a, new_b
                elif a.is_adopter and b.is_adopter:
                    # 両方とも導入者：相互に意見を更新（間接的にFundamental上昇の影響を受ける）
                    if abs(a.expressed_opinion - b.expressed_opinion) < self.confidence_threshold:
                        new_a = (1 - self.learning_rate) * a.expressed_opinion + self.learning_rate * b.expressed_opinion
                        new_b = (1 - self.learning_rate) * b.expressed_opinion + self.learning_rate * a.expressed_opinion
                        a.expressed_opinion, b.expressed_opinion = new_a, new_b

        self.tick += 1
        self._record()

    # ------------------------------- logging --------------------------------
    def _record(self):
        know_agents = [t for t in self.turtles if t.know == 1]
        self.sum_opinion.append(sum(t.expressed_opinion for t in know_agents if t.expressed_opinion is not None))
        self.know_count.append(len(know_agents))
        # 新しい指標: Aware かつ expressed_opinion > 0.5 のエージェント数
        aware_positive_agents = [t for t in know_agents if t.expressed_opinion is not None and t.expressed_opinion > 0.5]
        self.aware_positive_count.append(len(aware_positive_agents))
        # 導入者数を記録
        adopters = [t for t in self.turtles if t.adopted]
        self.adopter_count.append(len(adopters))
        self.opinion_distribution.append([t.expressed_opinion for t in know_agents if t.expressed_opinion is not None])
        self.ticks.append(self.tick)

    # ------------------------------- plotting -------------------------------
    
    def run(self, ticks: int = 400):
        self.setup()
        for _ in range(ticks):
            self.step()

    def plot_aware_positive_trend(self):
        """Aware かつ opinion > 0.5 のエージェント数の推移をプロット"""
        plt.figure(figsize=(3.5, 4))
        plt.plot(self.ticks, self.aware_positive_count, 'b-', linewidth=2, label='Aware Agents (opinion > 0.5)')
        
        plt.xlabel('Time (ticks)', fontsize=12)
        plt.ylabel('Number of Agents', fontsize=12)
        plt.title('Evolution of Aware Agents with Positive Opinion (> 0.5)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 最終値を表示
        final_aware_positive = self.aware_positive_count[-1] if self.aware_positive_count else 0
        
        plt.text(0.02, 0.98, 
                f'Final: {final_aware_positive} aware agents with opinion > 0.5',
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8), va='top')
        
        plt.tight_layout()
        #plt.savefig('aware_positive_opinion_trend.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_adopter_trend(self):
        """新技術導入者数の推移をプロット"""
        plt.figure(figsize=(3.5, 4))
        plt.plot(self.ticks, self.adopter_count, 'r-', linewidth=2, label='Technology Adopters')
        
        plt.xlabel('Time (ticks)', fontsize=12)
        plt.ylabel('Number of Adopters', fontsize=12)
        plt.title('Evolution of Technology Adopters', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 最終値を表示
        final_adopters = self.adopter_count[-1] if self.adopter_count else 0
        
        plt.text(0.02, 0.98, 
                f'Final: {final_adopters} technology adopters',
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8), va='top')
        
        plt.tight_layout()
        plt.show()

    def plot_sum_opinions_trend(self):
        """意見の総和の推移をプロット"""
        plt.figure(figsize=(3.5, 4))
        plt.plot(self.ticks, self.sum_opinion, 'b-', linewidth=2, label='Sum of Expressed Opinions')
        
        plt.xlabel('Time (ticks)', fontsize=12)
        plt.ylabel('Sum of Expressed Opinions', fontsize=12)
        plt.title('Evolution of Sum of Expressed Opinions', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 最終値を表示
        final_sum = self.sum_opinion[-1] if self.sum_opinion else 0
        
        plt.text(0.02, 0.98, 
                f'Final sum: {final_sum:.2f}',
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8), va='top')
        
        plt.tight_layout()
        plt.show()

# 複数シミュレーションの実行と平均化（導入者を含む）
def run_multiple_simulations_with_adoption(n_runs: int = 100, ticks: int = 250,
                                         *, agent_count: int = 999, learning_rate: float = 0.3,
                                         confidence_threshold: float = 0.5, adoption_probability: float = 0.01):
    all_aware_positive_counts = []
    all_adopter_counts = []
    all_sum_opinions = []
    
    print(f"Running {n_runs} simulations, {ticks} ticks each...")
    
    for i in range(n_runs):
        print(f"Simulation {i+1}/{n_runs}...", end="", flush=True)
        model = Model(agent_count=agent_count,
                      learning_rate=learning_rate,
                      confidence_threshold=confidence_threshold,
                      adoption_probability=adoption_probability)
        model.run(ticks=ticks)
        all_aware_positive_counts.append(model.aware_positive_count)
        all_adopter_counts.append(model.adopter_count)
        all_sum_opinions.append(model.sum_opinion)
        print(" done")

    print("All simulations completed. Creating plots...")
    
    # 平均と信頼区間の計算
    all_aware_positive_counts = np.array(all_aware_positive_counts)
    all_adopter_counts = np.array(all_adopter_counts)
    all_sum_opinions = np.array(all_sum_opinions)
    
    mean_aware_positive = np.mean(all_aware_positive_counts, axis=0)
    std_aware_positive = np.std(all_aware_positive_counts, axis=0)
    ci_aware_positive = 1.96 * std_aware_positive / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_aware_positive)
    
    mean_adopters = np.mean(all_adopter_counts, axis=0)
    std_adopters = np.std(all_adopter_counts, axis=0)
    ci_adopters = 1.96 * std_adopters / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_adopters)
    
    mean_sum_opinions = np.mean(all_sum_opinions, axis=0)
    std_sum_opinions = np.std(all_sum_opinions, axis=0)
    ci_sum_opinions = 1.96 * std_sum_opinions / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_sum_opinions)
    
    # プロット（3つの図）
    t = np.arange(ticks + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # 図1: Sum of Expressed Opinions
    ax1.plot(t, mean_sum_opinions, color='blue', label=f'Average over {n_runs} runs')
    if n_runs > 1:
        ax1.fill_between(t, mean_sum_opinions - ci_sum_opinions, mean_sum_opinions + ci_sum_opinions, 
                        alpha=0.1, color='blue', label='95% CI')
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Sum of Expressed Opinions', fontsize=12)
    ax1.set_xlim(left=0, right=ticks+1)
    ax1.set_ylim(bottom=0, top=max(75, np.max(mean_sum_opinions) * 1.1))
    ax1.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    ax1.set_xticks(np.arange(0, ticks+1, 50))
    ax1.text(-0.2, 1.1, '(a)', transform=ax1.transAxes,
             fontsize=15, fontweight='bold', va='top', ha='left')
    
    # 図2: Advocates
    ax2.plot(t, mean_aware_positive, color='green', label=f'Average over {n_runs} runs')
    if n_runs > 1:
        ax2.fill_between(t, mean_aware_positive - ci_aware_positive, mean_aware_positive + ci_aware_positive, 
                        alpha=0.1, color='green', label='95% CI')
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Number of Advocates', fontsize=12)
    ax2.set_xlim(left=0, right=ticks+1)
    ax2.set_ylim(bottom=0, top=75)
    ax2.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    ax2.set_xticks(np.arange(0, ticks+1, 50))  
    ax2.set_yticks(np.arange(0, 75, 20)) 
    ax2.text(-0.2, 1.1, '(b)', transform=ax2.transAxes,
             fontsize=15, fontweight='bold', va='top', ha='left')
    
    # 図3: Adopters
    ax3.plot(t, mean_adopters, color='red', label=f'Average over {n_runs} runs')
    if n_runs > 1:
        ax3.fill_between(t, mean_adopters - ci_adopters, mean_adopters + ci_adopters, 
                        alpha=0.1, color='red', label='95% CI')
    
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Number of Adopters', fontsize=12)
    ax3.set_xlim(left=0, right=ticks+1)
    ax3.set_ylim(bottom=0, top=max(50, np.max(mean_adopters) * 1.1))
    ax3.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    ax3.set_xticks(np.arange(0, ticks+1, 50))
    ax3.text(-0.2, 1.1, '(c)', transform=ax3.transAxes,
             fontsize=15, fontweight='bold', va='top', ha='left')
    
    plt.tight_layout()
    plt.savefig('adoption_expectation_coevolution_three_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mean_sum_opinions, mean_aware_positive, mean_adopters

# 4×3グリッドプロット：Very Fast, Fast, Slow, Very Slow の adoption_probability を比較
def run_adoption_speed_comparison(n_runs: int = 100, ticks: int = 300,
                                *, agent_count: int = 999, learning_rate: float = 0.3,
                                confidence_threshold: float = 0.5):
    # 4つのシナリオ（シナリオごとのticks数も設定）
    scenarios = {
        'Very Fast': {'prob': 0.05, 'ticks': ticks},
        'Fast': {'prob': 0.01, 'ticks': ticks}, 
        'Slow': {'prob': 0.001, 'ticks': 600},
        'Very Slow': {'prob': 0.0001, 'ticks': 600}
    }
    
    results = {}
    
    for scenario_name, config in scenarios.items():
        adoption_prob = config['prob']
        scenario_ticks = config['ticks']
        print(f"\nRunning {scenario_name} scenario (adoption_probability={adoption_prob}, ticks={scenario_ticks})...")
        all_sum_opinions = []
        all_aware_positive_counts = []
        all_adopter_counts = []
        
        for i in range(n_runs):
            print(f"  Simulation {i+1}/{n_runs}...", end="", flush=True)
            model = Model(agent_count=agent_count,
                          learning_rate=learning_rate,
                          confidence_threshold=confidence_threshold,
                          adoption_probability=adoption_prob)
            model.run(ticks=scenario_ticks)
            all_sum_opinions.append(model.sum_opinion)
            all_aware_positive_counts.append(model.aware_positive_count)
            all_adopter_counts.append(model.adopter_count)
            print(" done")
        
        # 平均と信頼区間を計算
        all_sum_opinions = np.array(all_sum_opinions)
        all_aware_positive_counts = np.array(all_aware_positive_counts)
        all_adopter_counts = np.array(all_adopter_counts)
        
        # Adoptersを％に変換（総エージェント数で割って100倍）
        total_agents = agent_count + 1  # +1 for developer
        all_adopter_percentages = (all_adopter_counts / total_agents) * 100
        
        results[scenario_name] = {
            'sum_opinions': {
                'mean': np.mean(all_sum_opinions, axis=0),
                'ci': 1.96 * np.std(all_sum_opinions, axis=0) / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(np.mean(all_sum_opinions, axis=0))
            },
            'advocates': {
                'mean': np.mean(all_aware_positive_counts, axis=0),
                'ci': 1.96 * np.std(all_aware_positive_counts, axis=0) / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(np.mean(all_aware_positive_counts, axis=0))
            },
            'adopters': {
                'mean': np.mean(all_adopter_percentages, axis=0),
                'ci': 1.96 * np.std(all_adopter_percentages, axis=0) / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(np.mean(all_adopter_percentages, axis=0))
            },
            'ticks': scenario_ticks
        }
    
    print("\nCreating 4x3 comparison plot...")
    
    # 4×3グリッドプロット作成
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    
    # 各行が指標、各列がシナリオ
    metrics = ['sum_opinions', 'advocates', 'adopters']
    metric_labels = ['Sum of Expressed Opinions', 'Number of Advocates', 'Percentage of Adopters (%)']
    colors = ['blue', 'green', 'red']
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        for j, (scenario, config) in enumerate(scenarios.items()):
            ax = axes[i, j]
            scenario_ticks = results[scenario]['ticks']
            t = np.arange(scenario_ticks + 1)  # シナリオごとの時間軸
            data_mean = results[scenario][metric]['mean']
            data_ci = results[scenario][metric]['ci']
            
            # 平均線をプロット
            ax.plot(t, data_mean, color=color, linewidth=2)
            
            # 信頼区間をプロット（透明度0.1に変更）
            if n_runs > 1:
                ax.fill_between(t, data_mean - data_ci, data_mean + data_ci, 
                               alpha=0.1, color=color)
            
            ax.set_xlim(0, scenario_ticks)  # シナリオごとの時間範囲
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # タイトルと軸ラベル（probを省略）
            if i == 0:  # 最上行にシナリオ名
                ax.set_title(f'{scenario} adoption', fontsize=14, fontweight='bold')
            if j == 0:  # 最左列に指標名
                ax.set_ylabel(label, fontsize=12)
            if i == 2:  # 最下行にx軸ラベル
                ax.set_xlabel('Time Step (t)', fontsize=12)
            
            # y軸の範囲を調整（全列で統一）
            if metric == 'sum_opinions':
                ax.set_ylim(0, 250)  # 全列で統一した固定範囲0-250
            elif metric == 'advocates':
                ax.set_ylim(0, 55)  # 固定範囲0-55（見切れ防止）
                ax.set_yticks(np.arange(0, 56, 20))  # 20刻み
            else:  # adopters (%)
                ax.set_ylim(0, 95) 
                ax.set_yticks(np.arange(0, 96, 20))  # 20刻み

    
    plt.tight_layout()
    plt.savefig('adoption_speed_comparison_4x3.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

# 新しい関数：awareなエージェントの意見の総和とadvocateの数を並べてプロット
def run_multiple_simulations_combined_plots(n_runs: int = 100, ticks: int = 250,
                                          *, agent_count: int = 999, learning_rate: float = 0.3,
                                          confidence_threshold: float = 0.5):
    all_sum_opinions = []
    all_aware_positive_counts = []
    
    print(f"Running {n_runs} simulations, {ticks} ticks each...")
    
    for i in range(n_runs):
        print(f"Simulation {i+1}/{n_runs}...", end="", flush=True)
        model = Model(agent_count=agent_count,
                      learning_rate=learning_rate,
                      confidence_threshold=confidence_threshold)
        model.run(ticks=ticks)
        all_sum_opinions.append(model.sum_opinion)
        all_aware_positive_counts.append(model.aware_positive_count)
        print(" done")

    print("All simulations completed. Creating combined plot...")
    
    # 平均と信頼区間の計算
    all_sum_opinions = np.array(all_sum_opinions)
    all_aware_positive_counts = np.array(all_aware_positive_counts)
    
    mean_sum_opinion = np.mean(all_sum_opinions, axis=0)
    std_sum_opinion = np.std(all_sum_opinions, axis=0)
    ci_sum_opinion = 1.96 * std_sum_opinion / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_sum_opinion)
    
    mean_aware_positive = np.mean(all_aware_positive_counts, axis=0)
    std_aware_positive = np.std(all_aware_positive_counts, axis=0)
    ci_aware_positive = 1.96 * std_aware_positive / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_aware_positive)
    
    # 並列プロット
    t = np.arange(ticks + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
    
    # 図(a): awareなエージェントの意見の総和
    ax1.plot(t, mean_sum_opinion, color='blue', label=f'Average over {n_runs} runs')
    if n_runs > 1:
        ax1.fill_between(t, mean_sum_opinion - ci_sum_opinion, mean_sum_opinion + ci_sum_opinion, 
                        alpha=0.1, color='blue', label='95% CI')
    
    ax1.set_xlabel('Time Step (t)', fontsize=12)
    ax1.set_ylabel('Sum of Expressed Opinions', fontsize=12)
    ax1.set_xlim(left=0, right=ticks+1)
    ax1.set_ylim(bottom=0, top=75)
    ax1.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    ax1.set_xticks(np.arange(0, ticks+1, 50))
    ax1.set_yticks(np.arange(0, 75, 20))
    ax1.text(-0.22, 1.1, '(a)', transform=ax1.transAxes,
             fontsize=15, fontweight='bold', va='top', ha='left')
    
    # 図(b): advocateの数
    ax2.plot(t, mean_aware_positive, color='green', label=f'Average over {n_runs} runs')
    if n_runs > 1:
        ax2.fill_between(t, mean_aware_positive - ci_aware_positive, mean_aware_positive + ci_aware_positive, 
                        alpha=0.1, color='green', label='95% CI')
    
    ax2.set_xlabel('Time Step (t)', fontsize=12)
    ax2.set_ylabel('Number of Advocates', fontsize=12)
    ax2.set_xlim(left=0, right=ticks+1)
    ax2.set_ylim(bottom=0, top=50)
    ax2.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    ax2.set_xticks(np.arange(0, ticks+1, 50))
    ax2.set_yticks(np.arange(0, 50, 20))
    ax2.text(-0.22, 1.1, '(b)', transform=ax2.transAxes,
             fontsize=15, fontweight='bold', va='top', ha='left')
    
    plt.tight_layout()
    plt.savefig('reproducing_hype_cycle_1.png', dpi=300, bbox_inches='tight')
    plt.savefig ('reproducing_hype_cycle_1.svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mean_sum_opinion, mean_aware_positive

if __name__ == '__main__':
    
    # 4×3グリッド比較プロット（Very Fast, Fast, Slow, Very Slow）
    print("Running adoption speed comparison (4x3 grid)...")
    run_adoption_speed_comparison(n_runs=100, ticks=300, agent_count=999)
    
    # 複数シミュレーション実行例（新機能付き）
    #print("\nRunning multiple simulations with adoption mechanism...")
    #run_multiple_simulations_with_adoption(n_runs=10, ticks=250, agent_count=999, adoption_probability=0.005)
    
    # 従来の並列図の実行例（比較のため）
    #print("\nRunning multiple simulations for combined plots...")
    #run_multiple_simulations_combined_plots(n_runs=100, ticks=200, agent_count=999)