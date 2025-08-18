import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

# -----------------------------------------------------------------------------
#  Agent definition
# -----------------------------------------------------------------------------

class Turtle:
    """Single agent with an opinion and a knowledge flag."""

    def __init__(self, latent_opinion: float = None, x: float | None = None, y: float | None = None):
        self.x = random.random() if x is None else x
        self.y = random.random() if y is None else y
        if latent_opinion is None:
            self.latent_opinion = np.random.normal(0, 0.5)
        else:
            self.latent_opinion = latent_opinion
        self.expressed_opinion = None  # None until agent becomes aware
        self.know = 0  # 0 = ignorant, 1 = knowledgeable

# -----------------------------------------------------------------------------
#  Opinion‑dynamics model
# -----------------------------------------------------------------------------

class Model:
    """Opinion dynamics with **no ignorant‑ignorant interactions**.

    ▸ One developer (know=1, opinion=variable) + `agent_count` ignorant agents.
    ▸ Only pairs that include at least ONE knowledgeable agent can interact.
    ▸ In know=1 vs know=0, only the ignorant side updates & gains knowledge.
    ▸ In know=1 vs know=1, both update.
    ▸ know=0 vs know=0 pairs are skipped entirely.
    """

    def __init__(self, agent_count: int = 999, *, learning_rate: float = 0.3, 
                 confidence_threshold: float = 0.5, developer_opinion: float = 1.0):
        self.agent_count = agent_count
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.developer_opinion = developer_opinion

        # runtime containers
        self.turtles = []
        self.tick = 0
        self.sum_opinion = []
        self.know_count = []
        self.opinion_distribution = []
        self.ticks = []

    # -------------------------------- setup ---------------------------------
    def setup(self):
        self.turtles.clear()
        self.tick = 0
        self.sum_opinion.clear()
        self.know_count.clear()
        self.opinion_distribution.clear()
        self.ticks.clear()

        for _ in range(self.agent_count):
            t = Turtle()
            self.turtles.append(t)

        developer = Turtle(latent_opinion=self.developer_opinion)
        developer.expressed_opinion = self.developer_opinion
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
        self.opinion_distribution.append([t.expressed_opinion for t in know_agents if t.expressed_opinion is not None])
        self.ticks.append(self.tick)

    # ------------------------------- running -------------------------------
    def run(self, ticks: int = 400):
        self.setup()
        for _ in range(ticks):
            self.step()

# -----------------------------------------------------------------------------
#  Multiple‑simulation helper with varying developer opinions
# -----------------------------------------------------------------------------

def compare_developer_opinions(n_runs: int = 100, ticks: int = 200, 
                             *, agent_count: int = 999, learning_rate: float = 0.3,
                             confidence_threshold: float = 0.5, 
                             developer_opinions: list = [-0.5, 0, 0.5, 1, 1.5],
                             extended_ticks: dict = None):
    """異なる開発者の意見値での結果を比較（上段：Sum of Opinions、下段：Number of Advocates）"""
    
    # デフォルトのextended_ticksを設定
    if extended_ticks is None:
        extended_ticks = {}
    
    # サブプロットのグリッドを作成（2行5列）
    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8))
    
    # Y軸範囲計算用のデータを保存
    all_sum_data = []
    all_advocate_data = []
    
    # 各開発者意見値についてシミュレーション
    for col_idx, dev_opinion in enumerate(developer_opinions):
        if col_idx >= ncols:  # 5列を超えないようにする
            break
        
        # この開発者意見値に対して延長したティック数があるか確認
        current_ticks = extended_ticks.get(dev_opinion, ticks)
            
        all_sums = []
        all_advocates = []
        
        for _ in range(n_runs):
            model = Model(agent_count=agent_count,
                         learning_rate=learning_rate,
                         confidence_threshold=confidence_threshold,
                         developer_opinion=dev_opinion)
            model.run(ticks=current_ticks)
            all_sums.append(model.sum_opinion)
            
            # aware かつ expressed_opinion > 0.5 のエージェント数を計算
            advocates_count = []
            for tick_opinions in model.opinion_distribution:
                advocates = sum(1 for opinion in tick_opinions if opinion is not None and opinion > 0.5)
                advocates_count.append(advocates)
            all_advocates.append(advocates_count)
        
        # 平均と信頼区間を計算
        all_sums = np.array(all_sums)
        all_advocates = np.array(all_advocates)
        
        mean_sum = np.mean(all_sums, axis=0)
        std_sum = np.std(all_sums, axis=0)
        ci_sum = 1.96 * std_sum / np.sqrt(n_runs)
        
        mean_advocates = np.mean(all_advocates, axis=0)
        std_advocates = np.std(all_advocates, axis=0)
        ci_advocates = 1.96 * std_advocates / np.sqrt(n_runs)
        
        # Y軸範囲計算用にデータを保存
        all_sum_data.extend([mean_sum - ci_sum, mean_sum + ci_sum])
        all_advocate_data.extend([mean_advocates - ci_advocates, mean_advocates + ci_advocates])
        
        t = np.arange(current_ticks + 1)
        
        # 上段: Sum of Opinions
        ax_sum = axes[0, col_idx]
        ax_sum.plot(t, mean_sum, color='blue', linewidth=2)
        ax_sum.fill_between(t, mean_sum - ci_sum, mean_sum + ci_sum, color='blue', alpha=0.2)
        ax_sum.set_title(f"Developer's Opinion : {dev_opinion}", fontsize=16)
        ax_sum.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
        
        # パネルラベル（上段：a-e）
        panel_label_upper = chr(ord('a') + col_idx)
        ax_sum.text(-0.02, 1.01, f'({panel_label_upper})', transform=ax_sum.transAxes, 
                   fontsize=16, fontweight='bold', va='bottom', ha='right')
        
        # Y軸ラベルは最左列のみ
        if col_idx == 0:
            ax_sum.set_ylabel('Sum of Expressed Opinions', fontsize=14)
        
        # 下段: Number of Advocates
        ax_advocates = axes[1, col_idx]
        ax_advocates.plot(t, mean_advocates, color='green', linewidth=2)
        ax_advocates.fill_between(t, mean_advocates - ci_advocates, mean_advocates + ci_advocates, color='green', alpha=0.2)
        ax_advocates.set_xlabel('Time Step (t)', fontsize=14)
        ax_advocates.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
        
        # パネルラベル（下段：f-j）
        panel_label_lower = chr(ord('f') + col_idx)
        ax_advocates.text(-0.02, 1.01, f'({panel_label_lower})', transform=ax_advocates.transAxes, 
                         fontsize=16, fontweight='bold', va='bottom', ha='right')
        
        # Y軸ラベルは最左列のみ
        if col_idx == 0:
            ax_advocates.set_ylabel('Number of Advocates', fontsize=14)
    
    # Y軸範囲を統一
    if all_sum_data:
        sum_min = np.min([np.min(data) for data in all_sum_data])
        sum_max = np.max([np.max(data) for data in all_sum_data])
        for col_idx in range(len(developer_opinions)):
            axes[0, col_idx].set_ylim(sum_min, sum_max)
            # Y軸の目盛りを適切な間隔に設定（Sum of Opinionsの場合）
            range_sum = sum_max - sum_min
            if range_sum > 300:
                interval = 50  # 範囲が大きい場合は100間隔
            elif range_sum > 150:
                interval = 30   # 中程度の場合は50間隔
            else:
                interval = 25   # 範囲が小さい場合は25間隔
            axes[0, col_idx].yaxis.set_major_locator(ticker.MultipleLocator(interval))

    
    if all_advocate_data:
        advocate_min = np.min([np.min(data) for data in all_advocate_data])
        advocate_max = np.max([np.max(data) for data in all_advocate_data])
        for col_idx in range(len(developer_opinions)):
            axes[1, col_idx].set_ylim(advocate_min, advocate_max)
            # Y軸の目盛りを30程度の間隔に設定
            axes[1, col_idx].yaxis.set_major_locator(ticker.MultipleLocator(30))
    
    # 余分な列を非表示に
    for col_idx in range(len(developer_opinions), ncols):
        axes[0, col_idx].axis('off')
        axes[1, col_idx].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.06, right=0.98)
    plt.savefig(f'developer_opinion_comparison_mu{learning_rate}_d{confidence_threshold}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'developer_opinion_comparison_mu{learning_rate}_d{confidence_threshold}.svg', dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------------------------------------------------------
#  Final opinion distribution comparison
# -----------------------------------------------------------------------------

def compare_final_distributions(n_runs: int = 30, ticks: int = 200,
                              *, agent_count: int = 999, learning_rate: float = 0.3,
                              confidence_threshold: float = 0.5, 
                              developer_opinions: list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75],
                              extended_ticks: dict = None):
    """異なる開発者意見値での最終的な意見分布を比較"""
    
    # デフォルトのextended_ticksを設定
    if extended_ticks is None:
        extended_ticks = {1.75: 300}  # 1.75の場合は300ティック
    
    # サブプロットのグリッド（2行4列）
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, dev_opinion in enumerate(developer_opinions):
        if idx >= len(axes):
            break
        
        # この開発者意見値に対して延長したティック数があるか確認
        current_ticks = extended_ticks.get(dev_opinion, ticks)
            
        all_final_opinions = []
        
        for _ in range(n_runs):
            model = Model(agent_count=agent_count,
                         learning_rate=learning_rate,
                         confidence_threshold=confidence_threshold,
                         developer_opinion=dev_opinion)
            model.run(ticks=current_ticks)
            # 最終的な意見分布を収集
            final_opinions = [t.expressed_opinion for t in model.turtles if t.know == 1 and t.expressed_opinion is not None]
            all_final_opinions.extend(final_opinions)
        
        # グリッドの位置を計算
        row = idx // ncols
        col = idx % ncols
        
        # ヒストグラムを作成
        ax = axes[idx]
        ax.hist(all_final_opinions, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        
        # X軸ラベルは最下行のみ表示
        if row == nrows - 1 or idx == len(developer_opinions) - 1:
            ax.set_xlabel('Opinion Value', fontsize=10)
        
        # Y軸ラベルは各列の最左端のみ表示
        if col == 0:
            ax.set_ylabel('Frequency', fontsize=10)
        
        # タイトル設定（ティック数も表示）
        tick_info = f" (ticks={current_ticks})" if current_ticks != ticks else ""
        ax.set_title(f"Developer Opinion: {dev_opinion}{tick_info}", fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        
        # X軸とY軸の範囲を統一
        ax.set_xlim(-0.5, 1.5)
    
    # 余分なサブプロットを非表示に
    for i in range(len(developer_opinions), len(axes)):
        axes[i].axis('off')
    
    # 共通の軸ラベル（中央に配置）
    fig.text(0.5, 0.01, 'Opinion Value', fontsize=12, ha='center')
    fig.text(0.01, 0.5, 'Frequency', fontsize=12, va='center', rotation='vertical')
    
    plt.suptitle(f'Final Opinion Distributions (μ={learning_rate}, d={confidence_threshold}, n={n_runs})', 
                fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.98)
    plt.savefig(f'final_opinion_distributions_mu{learning_rate}_d={confidence_threshold}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'final_opinion_distributions_mu{learning_rate}_d={confidence_threshold}.svg', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':

    # 様々な開発者意見値でのシミュレーション比較
    compare_developer_opinions(n_runs=100, ticks=200, 
                             developer_opinions=[-0.5, 0, 0.5, 1, 1.5])
    
    # 様々な開発者意見値での最終意見分布の比較
    #compare_final_distributions(n_runs=10, ticks=200, 
                              #developer_opinions=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75],
                              #extended_ticks=extended_ticks)