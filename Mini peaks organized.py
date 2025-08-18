import numpy as np
import matplotlib.pyplot as plt
import random

# -----------------------------------------------------------------------------
#  Agent definition
# -----------------------------------------------------------------------------

random.seed(42)  # for reproducibility
np.random.seed(42)  # for reproducibility   

class Turtle:
    """Single agent with an opinion and a knowledge flag."""

    def __init__(self, x: float | None = None, y: float | None = None, latent_opinion: float | None = None):
        self.x = random.random() if x is None else x
        self.y = random.random() if y is None else y
        self.latent_opinion = np.random.normal(0, 0.5) if latent_opinion is None else latent_opinion
        self.expressed_opinion = None  # only set when aware
        self.know = 0  # 0 = ignorant, 1 = knowledgeable
        self.color = None

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

    def __init__(self, agent_count: int = 999, *, learning_rate: float = 0.3, confidence_threshold: float = 0.5):
        self.agent_count = agent_count
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold

        # runtime containers
        self.turtles = []
        self.tick = 0
        self.sum_opinion = []
        self.know_count = []
        self.opinion_distribution = []
        self.ticks = []
        self.aware_opinion_sum = []  # 新しい指標: Aware agents の opinion の合計
        
        # 5つのグループのawareness rate追跡用（combined_figures_plot.pyと同様）
        self.medium_positive_ratio = []  # Potential moderate advocates (0.5 < latent_opinion < 1.0)
        self.strong_positive_ratio = []  # Potential extreme advocates (latent_opinion >= 1.0)
        self.neutral_ratio = []          # Potential neutral agents (-0.5 <= latent_opinion <= 0.5)
        self.moderate_opponents_ratio = []  # Potential moderate opponents (-1.0 < latent_opinion < -0.5)
        self.extreme_opponents_ratio = []   # Potential extreme opponents (latent_opinion <= -1.0)
        
        # 単純化のための合算用
        self.advocates_ratio = []  # All advocates combined (latent_opinion > 0.5)
        self.opponents_ratio = []  # All opponents combined (latent_opinion < -0.5)

    # -------------------------------- setup ---------------------------------
    def setup(self):
        self.turtles.clear()
        self.tick = 0
        self.sum_opinion.clear()
        self.know_count.clear()
        self.opinion_distribution.clear()
        self.ticks.clear()
        self.aware_opinion_sum.clear()
        self.medium_positive_ratio.clear()
        self.strong_positive_ratio.clear()
        self.neutral_ratio.clear()
        self.moderate_opponents_ratio.clear()
        self.extreme_opponents_ratio.clear()
        self.advocates_ratio.clear()
        self.opponents_ratio.clear()

        for _ in range(self.agent_count):
            self.turtles.append(Turtle())

        # Developer agent: always aware, latent_opinion=0.15, expressed_opinion=0.15
        developer = Turtle(latent_opinion=0.15)
        developer.expressed_opinion = 0.15
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

            if b.know == 0:
                # ignorant partner: similarity check with latent opinion
                if abs(a.expressed_opinion - b.latent_opinion) < self.confidence_threshold:
                    # update: b becomes aware, forms expressed opinion
                    b.expressed_opinion = (1 - self.learning_rate) * b.latent_opinion + self.learning_rate * a.expressed_opinion
                    b.know = 1
                # else: no update
            elif b.know == 1:
                # both aware: similarity check with expressed opinions
                if abs(a.expressed_opinion - b.expressed_opinion) < self.confidence_threshold:
                    new_a = (1 - self.learning_rate) * a.expressed_opinion + self.learning_rate * b.expressed_opinion
                    new_b = (1 - self.learning_rate) * b.expressed_opinion + self.learning_rate * a.expressed_opinion
                    a.expressed_opinion, b.expressed_opinion = new_a, new_b
                # else: no update
            # no else: a is knowledgeable by definition

        self.tick += 1
        self._record()

    # ------------------------------- logging --------------------------------
    def _record(self):
        know_agents = [t for t in self.turtles if t.know == 1]
        self.sum_opinion.append(sum(t.expressed_opinion for t in know_agents if t.expressed_opinion is not None))
        self.know_count.append(len(know_agents))
        # 新しい指標: Aware agents の expressed_opinion の合計
        aware_opinion_total = sum(t.expressed_opinion for t in know_agents if t.expressed_opinion is not None)
        self.aware_opinion_sum.append(aware_opinion_total)
        # 1. Potential moderate advocates (0.5 < latent_opinion < 1.0)
        potential_moderate_advocates = [t for t in self.turtles if 0.5 < t.latent_opinion < 1.0]
        actual_moderate_advocates = [t for t in potential_moderate_advocates if t.know == 1]
        self.medium_positive_ratio.append(len(actual_moderate_advocates) / max(len(potential_moderate_advocates), 1))
        # 2. Potential extreme advocates (latent_opinion >= 1.0)
        potential_extreme_advocates = [t for t in self.turtles if t.latent_opinion >= 1.0]
        actual_extreme_advocates = [t for t in potential_extreme_advocates if t.know == 1]
        self.strong_positive_ratio.append(len(actual_extreme_advocates) / max(len(potential_extreme_advocates), 1))
        # 3. Potential neutral agents (-0.5 <= latent_opinion <= 0.5)
        potential_neutral = [t for t in self.turtles if -0.5 <= t.latent_opinion <= 0.5]
        actual_neutral = [t for t in potential_neutral if t.know == 1]
        self.neutral_ratio.append(len(actual_neutral) / max(len(potential_neutral), 1))
        # 4. Potential moderate opponents (-1.0 < latent_opinion < -0.5)
        potential_moderate_opponents = [t for t in self.turtles if -1.0 < t.latent_opinion < -0.5]
        actual_moderate_opponents = [t for t in potential_moderate_opponents if t.know == 1]
        self.moderate_opponents_ratio.append(len(actual_moderate_opponents) / max(len(potential_moderate_opponents), 1))
        # 5. Potential extreme opponents (latent_opinion <= -1.0)
        potential_extreme_opponents = [t for t in self.turtles if t.latent_opinion <= -1.0]
        actual_extreme_opponents = [t for t in potential_extreme_opponents if t.know == 1]
        self.extreme_opponents_ratio.append(len(actual_extreme_opponents) / max(len(potential_extreme_opponents), 1))
        # 合算用：All advocates (latent_opinion > 0.5)
        potential_all_advocates = [t for t in self.turtles if t.latent_opinion > 0.5]
        actual_all_advocates = [t for t in potential_all_advocates if t.know == 1]
        self.advocates_ratio.append(len(actual_all_advocates) / max(len(potential_all_advocates), 1))
        # 合算用：All opponents (latent_opinion < -0.5)
        potential_all_opponents = [t for t in self.turtles if t.latent_opinion < -0.5]
        actual_all_opponents = [t for t in potential_all_opponents if t.know == 1]
        self.opponents_ratio.append(len(actual_all_opponents) / max(len(potential_all_opponents), 1))
        self.opinion_distribution.append([t.expressed_opinion for t in know_agents if t.expressed_opinion is not None])
        self.ticks.append(self.tick)

    # ------------------------------- plotting -------------------------------
    
    def run(self, ticks: int = 400):
        self.setup()
        for _ in range(ticks):
            self.step()

    def plot_aware_opinion_sum_trend(self):
        """Aware agents の opinion の合計の推移をプロット"""
        plt.figure(figsize=(3.5, 4))
        plt.plot(self.ticks, self.aware_opinion_sum, 'b-', alpha=0.7, linewidth=2, label='Sum of Aware Agents Opinion')
        
        plt.xlabel('Time Step (t)', fontsize=12)
        plt.ylabel('Sum of Opinions', fontsize=12)
        plt.xlim(left=0, right=len(self.ticks))
        plt.ylim(bottom=0, top=max(self.aware_opinion_sum) * 1.1 if self.aware_opinion_sum else 10)
        plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
        
        
        plt.tight_layout()
        plt.savefig('single_simulation_aware_opinion_sum.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_advocate_count_trend(self):
        """5つのグループのawareness rateの推移をプロット（詳細版）"""
        
        # スタイル設定
        STYLE = {
            "ext_adv": dict(color="skyblue", lw=3.5, alpha=1.0, zorder=5),
            "mod_adv": dict(color="skyblue", lw=2.5, ls="--", alpha=0.8, zorder=3),
            "neutral": dict(color="#A9B7C6", lw=2.5, ls=":", alpha=0.7, zorder=2),
            "ext_opp": dict(color="orange", lw=3.5, alpha=0.9, zorder=4),
            "mod_opp": dict(color="orange", lw=2.5, ls="--", alpha=0.8, zorder=3),
        }
        
        plt.figure(figsize=(6, 4))
        
        # 5つのグループをプロット
        plt.plot(self.ticks, [x * 100 for x in self.strong_positive_ratio], 
                label="Potential Extreme Advocates", **STYLE["ext_adv"])
        plt.plot(self.ticks, [x * 100 for x in self.medium_positive_ratio], 
                label="Potential Moderate Advocates", **STYLE["mod_adv"])
        plt.plot(self.ticks, [x * 100 for x in self.neutral_ratio], 
                label="Potential Neutral Agents", **STYLE["neutral"])
        plt.plot(self.ticks, [x * 100 for x in self.moderate_opponents_ratio], 
                label="Potential Moderate Opponents", **STYLE["mod_opp"])
        plt.plot(self.ticks, [x * 100 for x in self.extreme_opponents_ratio], 
                label="Potential Extreme Opponents", **STYLE["ext_opp"])
        
        plt.xlabel('Time Step (t)', fontsize=12)
        plt.ylabel('Awareness Rate (%)', fontsize=12)
        plt.xlim(left=0, right=len(self.ticks))
        plt.ylim(-2, 102)
        plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
        
        # レジェンド設定
        legend = plt.legend(loc="lower right", frameon=False, fancybox=False, shadow=False, 
                           framealpha=0.95, edgecolor='none', fontsize=10)
        legend.get_frame().set_facecolor('#FFFFFF')
        
        # 軸の設定
        plt.xticks(np.arange(0, len(self.ticks), 50))
        plt.yticks(np.arange(0, 101, 20))
        
        plt.tight_layout()
        plt.savefig('five_group_opinion_awareness_rates.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_combined_trends(self):
        """Sum of opinionsと5つのグループのawareness rateを並べて表示"""
        
        # スタイル設定
        STYLE = {
            "ext_adv": dict(color="skyblue", lw=3.5, alpha=1.0, zorder=5),
            "mod_adv": dict(color="skyblue", lw=2.5, ls="--", alpha=0.8, zorder=3),
            "neutral": dict(color="#A9B7C6", lw=2.5, ls=":", alpha=0.7, zorder=2),
            "ext_opp": dict(color="orange", lw=3.5, alpha=0.9, zorder=4),
            "mod_opp": dict(color="orange", lw=2.5, ls="--", alpha=0.8, zorder=3),
        }
        
        # 2つのサブプロットを作成（幅比1:2）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7), gridspec_kw={'width_ratios': [2, 3]})
        
        # 左側: Sum of Opinions
        ax1.plot(self.ticks, self.aware_opinion_sum, 'b-', alpha=0.7, linewidth=2, label='Sum of Aware Agents Opinion')
        ax1.set_xlabel('Time Step (t)', fontsize=14)
        ax1.set_ylabel('Sum of Expressed Opinions', fontsize=14)
        ax1.set_xlim(left=0, right=len(self.ticks))
        ax1.set_ylim(bottom=0, top=max(self.aware_opinion_sum) * 1.1 if self.aware_opinion_sum else 10)
        ax1.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
        ax1.text(-0.1, 1.06, '(a)', transform=ax1.transAxes,
                 fontsize=16, fontweight='bold', va='top', ha='center')
        
        # 右側: 5つのグループのAwareness Rate
        ax2.plot(self.ticks, [x * 100 for x in self.strong_positive_ratio], 
                label="Potential Extreme Advocates", **STYLE["ext_adv"])
        ax2.plot(self.ticks, [x * 100 for x in self.medium_positive_ratio], 
                label="Potential Moderate Advocates", **STYLE["mod_adv"])
        ax2.plot(self.ticks, [x * 100 for x in self.neutral_ratio], 
                label="Potential Neutral Agents", **STYLE["neutral"])
        ax2.plot(self.ticks, [x * 100 for x in self.moderate_opponents_ratio], 
                label="Potential Moderate Opponents", **STYLE["mod_opp"])
        ax2.plot(self.ticks, [x * 100 for x in self.extreme_opponents_ratio], 
                label="Potential Extreme Opponents", **STYLE["ext_opp"])
        
        ax2.set_xlabel('Time Step (t)', fontsize=14)
        ax2.set_ylabel('Awareness Rate (%)', fontsize=14)
        ax2.set_xlim(left=0, right=len(self.ticks))
        ax2.set_ylim(-2, 102)
        ax2.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
        ax2.text(-0.1, 1.06, '(b)', transform=ax2.transAxes,
                 fontsize=16, fontweight='bold', va='top', ha='center')
        
        # レジェンド設定
        legend = ax2.legend(loc="lower right", frameon=False, fancybox=False, shadow=False, 
                           framealpha=0.95, edgecolor='none', fontsize=9)
        legend.get_frame().set_facecolor('#FFFFFF')
        
        # 軸の設定
        ax1.set_xticks(np.arange(0, len(self.ticks), 50))
        ax2.set_xticks(np.arange(0, len(self.ticks), 50))
        ax2.set_yticks(np.arange(0, 101, 20))
        
        plt.tight_layout()
        plt.savefig('combined_opinion_sum_and_awareness_rates.png', dpi=300, bbox_inches='tight')
        plt.savefig('combined_opinion_sum_and_awareness_rates.svg', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    # 単一シミュレーション実行例
    print("Running single simulation...")
    model = Model(agent_count=999, learning_rate=0.3, confidence_threshold=0.25)
    model.run(ticks=250)

    # 個別のプロット（必要に応じて）
    # model.plot_aware_opinion_sum_trend()
    # model.plot_advocate_count_trend()
    
    # 結合プロット
    model.plot_combined_trends()