import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import skewnorm

# -----------------------------------------------------------------------------
#  Agent definition
# -----------------------------------------------------------------------------

class Turtle:
    """Single agent with an opinion and a knowledge flag."""

    def __init__(self, x: float | None = None, y: float | None = None, 
                 distribution_type: str = "normal", dist_params: dict = None, latent_opinion: float = None):
        self.x = random.random() if x is None else x
        self.y = random.random() if y is None else y
        if latent_opinion is not None:
            self.latent_opinion = latent_opinion
        else:
            if dist_params is None:
                dist_params = {}
            if distribution_type == "normal":
                mean = dist_params.get("mean", 0)
                std = dist_params.get("std", 0.5)
                self.latent_opinion = np.random.normal(mean, std)
            elif distribution_type == "uniform":
                low = dist_params.get("low", -1.0)
                high = dist_params.get("high", 1.0)
                self.latent_opinion = np.random.uniform(low, high)
            elif distribution_type == "skewnorm":
                skewness = dist_params.get("skewness", 0)  # 歪度パラメータ（負=左歪み、正=右歪み）
                target_mean = dist_params.get("target_mean", 0)  # 目標平均
                target_std = dist_params.get("target_std", 0.5)  # 目標標準偏差
                
                # 歪正規分布の理論的平均と標準偏差の公式に基づいて調整
                delta = skewness / np.sqrt(1 + skewness**2)
                mu_z = delta * np.sqrt(2 / np.pi)  # 標準歪正規分布の平均
                sigma_z = np.sqrt(1 - mu_z**2)  # 標準歪正規分布の標準偏差
                
                # 目標平均・標準偏差に合わせるためのスケールと位置調整
                scale = target_std / sigma_z if sigma_z > 0 else target_std
                loc = target_mean - scale * mu_z
                
                self.latent_opinion = skewnorm.rvs(skewness, loc=loc, scale=scale)
            else:
                self.latent_opinion = np.random.normal(0, 0.5)
        self.expressed_opinion = None  # None until agent becomes aware
        self.know = 0  # 0 = ignorant, 1 = knowledgeable
        self.color = None

    def refresh_color(self):
        op = self.expressed_opinion if self.expressed_opinion is not None else self.latent_opinion
        norm = (max(min(op, 1.0), -1.0) + 1) / 2
        self.color = plt.cm.viridis(norm)

# -----------------------------------------------------------------------------
#  Opinion‑dynamics model
# -----------------------------------------------------------------------------

class Model:
    """Opinion dynamics with **no ignorant‑ignorant interactions**.

    ▸ One believer (know=1, opinion=1.0) + `agent_count` ignorant agents.
    ▸ Only pairs that include at least ONE knowledgeable agent can interact.
    ▸ In know=1 vs know=0, only the ignorant side updates & gains knowledge.
    ▸ In know=1 vs know=1, both update.
    ▸ know=0 vs know=0 pairs are skipped entirely.
    """

    def __init__(self, agent_count: int = 999, *, learning_rate: float = 0.3, 
                 confidence_threshold: float = 0.5, distribution_type: str = "normal", 
                 dist_params: dict = None):
        self.agent_count = agent_count
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.distribution_type = distribution_type
        self.dist_params = dist_params if dist_params is not None else {}

        # runtime containers
        self.turtles = []
        self.tick = 0
        self.sum_opinion = []
        self.know_count = []
        self.opinion_distribution = []
        self.ticks = []
        self.aware_positive_count = []  # 新しい指標: Aware かつ opinion > 0.5 の数

    # -------------------------------- setup ---------------------------------
    def setup(self):
        self.turtles.clear()
        self.tick = 0
        self.sum_opinion.clear()
        self.know_count.clear()
        self.opinion_distribution.clear()
        self.ticks.clear()

        for _ in range(self.agent_count):
            t = Turtle(distribution_type=self.distribution_type, dist_params=self.dist_params)
            self.turtles.append(t)

        developer = Turtle(latent_opinion=1.0)
        developer.expressed_opinion = 1.0 # Change here for different initial opinion
        developer.know = 1
        self.turtles.append(developer)

        self._record()

    # -------------------------------- step ----------------------------------
    def step(self):
        knowledgeable_agents = [t for t in self.turtles if t.know == 1]
        random.shuffle(knowledgeable_agents)

        for a in knowledgeable_agents:
            candidates = [p for p in self.turtles if p is not a]
            b = random.choice(candidates)

            a_op = a.expressed_opinion if a.know == 1 else a.latent_opinion
            b_op = b.expressed_opinion if b.know == 1 else b.latent_opinion
            interact = (abs(a_op - b_op) < self.confidence_threshold)
            if not interact:
                continue

            if a.know == 1 and b.know == 0:
                b.expressed_opinion = b.latent_opinion + self.learning_rate * (a.expressed_opinion - b.latent_opinion)
                b.know = 1
            elif a.know == 1 and b.know == 1:
                new_a = a.expressed_opinion + self.learning_rate * (b.expressed_opinion - a.expressed_opinion)
                new_b = b.expressed_opinion + self.learning_rate * (a.expressed_opinion - b.expressed_opinion)
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

# 異なる分布による4つのパネル比較
def run_distribution_comparison(n_runs: int = 100, ticks: int = 400, 
                               *, agent_count: int = 999, learning_rate: float = 0.3, 
                               confidence_threshold: float = 0.5):
    """異なる初期意見分布による4つのパネル比較"""
    
    # 異なる分布設定
    distributions = {
        "Negatively-Skewed": {
            "type": "skewnorm",
            "params": {"skewness": -3, "target_mean": 0, "target_std": 0.5}
        },
        "Normal (Symmetric)": {
            "type": "normal",
            "params": {"mean": 0, "std": 0.5}
        },
        "Positively-Skewed": {
            "type": "skewnorm",
            "params": {"skewness": 3, "target_mean": 0, "target_std": 0.5}
        }
    }
    
    results = {}
    
    # 各分布でシミュレーション実行
    for dist_name, dist_config in distributions.items():
        print(f"\nRunning simulations for {dist_name}")
        all_sum_opinions = []
        all_aware_positive_counts = []
        
        print(f"Running {n_runs} simulations, {ticks} ticks each...")
        
        for i in range(n_runs):
            print(f"Simulation {i+1}/{n_runs}...", end="", flush=True)
            model = Model(agent_count=agent_count,
                          learning_rate=learning_rate,
                          confidence_threshold=confidence_threshold,
                          distribution_type=dist_config["type"],
                          dist_params=dist_config["params"])
            model.run(ticks=ticks)
            all_sum_opinions.append(model.sum_opinion)
            all_aware_positive_counts.append(model.aware_positive_count)
            print(" done")
        
        # 平均と信頼区間の計算
        all_sum_opinions = np.array(all_sum_opinions)
        all_aware_positive_counts = np.array(all_aware_positive_counts)
        
        mean_sum_opinion = np.mean(all_sum_opinions, axis=0)
        std_sum_opinion = np.std(all_sum_opinions, axis=0)
        ci_sum_opinion = 1.96 * std_sum_opinion / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_sum_opinion)
        
        mean_aware_positive = np.mean(all_aware_positive_counts, axis=0)
        std_aware_positive = np.std(all_aware_positive_counts, axis=0)
        ci_aware_positive = 1.96 * std_aware_positive / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_aware_positive)
        
        results[dist_name] = {
            'mean_sum_opinion': mean_sum_opinion,
            'ci_sum_opinion': ci_sum_opinion,
            'mean_aware_positive': mean_aware_positive,
            'ci_aware_positive': ci_aware_positive
        }
    
    print("\nAll simulations completed. Creating distribution comparison plot...")
    
    # 9パネルプロット作成（3×3：初期分布、意見の総和、advocate数）
    t = np.arange(ticks + 1)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    dist_names = list(distributions.keys())
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    
    # 各分布の結果をプロット
    for j, dist_name in enumerate(dist_names):
        # 上段：初期分布
        ax_dist = axes[0, j]
        x = np.linspace(-2, 2, 1000)
        
        if distributions[dist_name]["type"] == "normal":
            mean = distributions[dist_name]["params"]["mean"]
            std = distributions[dist_name]["params"]["std"]
            y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        elif distributions[dist_name]["type"] == "skewnorm":
            skewness = distributions[dist_name]["params"]["skewness"]
            target_mean = distributions[dist_name]["params"]["target_mean"]
            target_std = distributions[dist_name]["params"]["target_std"]
            
            # Calculate adjusted parameters (same logic as in Turtle class)
            delta = skewness / np.sqrt(1 + skewness**2)
            mu_z = delta * np.sqrt(2 / np.pi)
            sigma_z = np.sqrt(1 - mu_z**2)
            
            scale = target_std / sigma_z if sigma_z > 0 else target_std
            loc = target_mean - scale * mu_z
            
            y = skewnorm.pdf(x, skewness, loc=loc, scale=scale)
        
        ax_dist.plot(x, y, 'gray', linewidth=2)
        ax_dist.fill_between(x, y, alpha=0.3, color='lightgray')
        # 記号を左上に配置
        ax_dist.text(0,1.01, f'{letters[j]}.', fontsize=14, fontweight='bold', 
                     transform=ax_dist.transAxes, verticalalignment='bottom')
        # 分布名を中央に配置
        ax_dist.set_title(dist_name, fontsize=14, fontweight='bold', loc='center')
        ax_dist.set_xlabel('Latent Opinion', fontsize=12)
        ax_dist.set_ylabel('Density', fontsize=12)
        ax_dist.set_xlim(-2, 2)
        ax_dist.grid(True, alpha=0.3)
        
        # 中段：意見の総和
        ax_sum = axes[1, j]
        ax_sum.plot(t, results[dist_name]['mean_sum_opinion'], color='blue', linewidth=2)
        if n_runs > 1:
            ax_sum.fill_between(t, 
                               results[dist_name]['mean_sum_opinion'] - results[dist_name]['ci_sum_opinion'], 
                               results[dist_name]['mean_sum_opinion'] + results[dist_name]['ci_sum_opinion'], 
                               alpha=0.1, color='blue')
        
        ax_sum.set_xlabel('Time Step (t)', fontsize=12)
        ax_sum.set_ylabel('Sum of Expressed Opinions', fontsize=12)
        ax_sum.set_xlim(left=0, right=ticks+1)
        min_val = np.min(results[dist_name]['mean_sum_opinion'] - results[dist_name]['ci_sum_opinion'])
        max_val = np.max(results[dist_name]['mean_sum_opinion'] + results[dist_name]['ci_sum_opinion'])
        margin = (max_val - min_val) * 0.1 if max_val != min_val else 10
        ax_sum.set_ylim(bottom=min_val - margin, top=max_val + margin)
        ax_sum.grid(True, alpha=0.3)
        ax_sum.set_title(f'{letters[j+3]}.', fontsize=14, fontweight='bold', loc='left')
        
        # 下段：Advocate数
        ax_advocates = axes[2, j]
        ax_advocates.plot(t, results[dist_name]['mean_aware_positive'], color='green', linewidth=2)
        if n_runs > 1:
            ax_advocates.fill_between(t, 
                                     results[dist_name]['mean_aware_positive'] - results[dist_name]['ci_aware_positive'], 
                                     results[dist_name]['mean_aware_positive'] + results[dist_name]['ci_aware_positive'], 
                                     alpha=0.1, color='green')
        
        ax_advocates.set_xlabel('Time Step (t)', fontsize=12)
        ax_advocates.set_ylabel('Number of Advocates', fontsize=12)
        ax_advocates.set_xlim(left=0, right=ticks+1)
        min_val_adv = np.min(results[dist_name]['mean_aware_positive'] - results[dist_name]['ci_aware_positive'])
        max_val_adv = np.max(results[dist_name]['mean_aware_positive'] + results[dist_name]['ci_aware_positive'])
        margin_adv = (max_val_adv - min_val_adv) * 0.1 if max_val_adv != min_val_adv else 10
        ax_advocates.set_ylim(bottom=min_val_adv - margin_adv, top=max_val_adv + margin_adv)
        ax_advocates.grid(True, alpha=0.3)
        ax_advocates.set_title(f'{letters[j+6]}.', fontsize=14, fontweight='bold', loc='left')
    
    plt.tight_layout()
    plt.savefig('distribution_comparison_mean_zero.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

# 初期分布の可視化関数
def plot_initial_distributions(n_samples: int = 10000):
    """異なる初期分布の形状を可視化"""
    
    distributions = {
        "Negatively-Skewed": {
            "type": "skewnorm",
            "params": {"skewness": -3, "target_mean": 0, "target_std": 0.5}
        },
        "Normal (Symmetric)": {
            "type": "normal",
            "params": {"mean": 0, "std": 0.5}
        },
        "Positively-Skewed": {
            "type": "skewnorm",
            "params": {"skewness": 3, "target_mean": 0, "target_std": 0.5}
        }
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (dist_name, dist_config) in enumerate(distributions.items()):
        # Create x values for plotting theoretical distributions
        x = np.linspace(-2, 2, 1000)
        
        # Calculate theoretical PDF based on distribution type
        if dist_config["type"] == "normal":
            mean = dist_config["params"]["mean"]
            std = dist_config["params"]["std"]
            y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        
        elif dist_config["type"] == "skewnorm":
            skewness = dist_config["params"]["skewness"]
            target_mean = dist_config["params"]["target_mean"]
            target_std = dist_config["params"]["target_std"]
            
            # Calculate adjusted parameters (same logic as in Turtle class)
            delta = skewness / np.sqrt(1 + skewness**2)
            mu_z = delta * np.sqrt(2 / np.pi)
            sigma_z = np.sqrt(1 - mu_z**2)
            
            scale = target_std / sigma_z if sigma_z > 0 else target_std
            loc = target_mean - scale * mu_z
            
            # Calculate theoretical PDF using scipy
            y = skewnorm.pdf(x, skewness, loc=loc, scale=scale)
        
        # Plot theoretical distribution
        axes[i].plot(x, y, 'b-', linewidth=2, label='Theoretical PDF')
        axes[i].fill_between(x, y, alpha=0.3, color='skyblue')
        axes[i].set_title(f'{dist_name}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Initial Opinion', fontsize=12)
        axes[i].set_ylabel('Density', fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(-2, 2)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('initial_distributions_mean_zero.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 初期分布の可視化
    print("Plotting initial distributions...")
    plot_initial_distributions()
    
    # 分布比較の実行
    print("\nRunning distribution comparison...")
    run_distribution_comparison(n_runs=100, ticks=250, agent_count=999)
    