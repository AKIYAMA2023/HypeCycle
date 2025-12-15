import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.gridspec import GridSpec


# -----------------------------------------------------------------------------
#  Agent definition
# -----------------------------------------------------------------------------

random.seed(42)  # for reproducibility
np.random.seed(42)

class Turtle:
    """Single agent with an opinion and a knowledge flag."""

    def __init__(self, x: float | None = None, y: float | None = None, latent_opinion: float | None = None):
        self.x = random.random() if x is None else x
        self.y = random.random() if y is None else y
        self.latent_opinion = np.random.normal(0, 0.5) if latent_opinion is None else latent_opinion  # fixed enthusiasm
        self.expressed_opinion = None  # evolving, only set when aware
        self.know = 0  # 0 = ignorant, 1 = knowledgeable
        self.color = None

    def refresh_color(self):
        val = self.expressed_opinion if self.know == 1 and self.expressed_opinion is not None else self.latent_opinion
        norm = (max(min(val, 1.0), -1.0) + 1) / 2
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

    # -------------------------------- setup ---------------------------------
    def setup(self):
        self.turtles.clear()
        self.tick = 0
        self.sum_opinion.clear()
        self.know_count.clear()
        self.opinion_distribution.clear()
        self.ticks.clear()

        for _ in range(self.agent_count):
            self.turtles.append(Turtle())

        # Developer agent: always aware, latent_opinion=1, expressed_opinion=1
        developer = Turtle(latent_opinion=1.0)
        developer.expressed_opinion = 1.0
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
        self.opinion_distribution.append([t.expressed_opinion for t in know_agents if t.expressed_opinion is not None])
        self.ticks.append(self.tick)

    # ------------------------------- plotting -------------------------------
    
    def run(self, ticks: int = 400):
        self.setup()
        for _ in range(ticks):
            self.step()

# -----------------------------------------------------------------------------
#  Multiple‑simulation helper 
# -----------------------------------------------------------------------------
 

    def run_simulation(self, ticks=250):
        self.setup()
        for _ in range(ticks):
            self.go()
        self.plot_opinion_sum_trend()
        self.plot_opinion_distribution()



def run_multiple_simulations(n_runs: int = 100, ticks: int = 250,
                            *, agent_count: int = 999, learning_rate: float = 0.3,
                            confidence_threshold: float = 0.5):
    all_sums = []
    for _ in range(n_runs):
        model = Model(agent_count=agent_count,
                      learning_rate=learning_rate,
                      confidence_threshold=confidence_threshold)
        model.run(ticks=ticks)
        all_sums.append(model.sum_opinion)

    all_sums = np.array(all_sums)
    mean = np.mean(all_sums, axis=0)
    std = np.std(all_sums, axis=0)
    ci = 1.96 * std / np.sqrt(n_runs)

    t = np.arange(ticks + 1)
    plt.figure(figsize=(3.5, 4))
    plt.plot(t, mean, label=f'Average over {n_runs} runs')
    plt.fill_between(t, mean - ci, mean + ci, alpha=0.1, label='95% CI')
    plt.xlabel('Time',fontsize=12)
    plt.ylabel('Sum of Expressed Opinions',fontsize=12)
    plt.text(-0.2, 1.1, '(a)', transform=plt.gca().transAxes,
         fontsize=15, fontweight='bold', va='top', ha='left')
    plt.xticks(np.arange(0, ticks+1, 50))  
    plt.yticks(np.arange(0, 75, 20)) 
    plt.xlim(left=0,right = ticks+1)
    plt.ylim(bottom=0,top=75)
    plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    plt.legend(loc='upper right', frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.gridspec import GridSpec


# -----------------------------------------------------------------------------
#  Agent definition
# -----------------------------------------------------------------------------

random.seed(42)  # for reproducibility
np.random.seed(42)

class Turtle:
    """Single agent with an opinion and a knowledge flag."""

    def __init__(self, x: float | None = None, y: float | None = None, latent_opinion: float | None = None):
        self.x = random.random() if x is None else x
        self.y = random.random() if y is None else y
        self.latent_opinion = np.random.normal(0, 0.5) if latent_opinion is None else latent_opinion  # fixed enthusiasm
        self.expressed_opinion = None  # evolving, only set when aware
        self.know = 0  # 0 = ignorant, 1 = knowledgeable
        self.color = None

    def refresh_color(self):
        val = self.expressed_opinion if self.know == 1 and self.expressed_opinion is not None else self.latent_opinion
        norm = (max(min(val, 1.0), -1.0) + 1) / 2
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

    # -------------------------------- setup ---------------------------------
    def setup(self):
        self.turtles.clear()
        self.tick = 0
        self.sum_opinion.clear()
        self.know_count.clear()
        self.opinion_distribution.clear()
        self.ticks.clear()

        for _ in range(self.agent_count):
            self.turtles.append(Turtle())

        # Developer agent: always aware, latent_opinion=1, expressed_opinion=1
        developer = Turtle(latent_opinion=1.0)
        developer.expressed_opinion = 1.0
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
        self.opinion_distribution.append([t.expressed_opinion for t in know_agents if t.expressed_opinion is not None])
        self.ticks.append(self.tick)

    # ------------------------------- plotting -------------------------------
    
    def run(self, ticks: int = 400):
        self.setup()
        for _ in range(ticks):
            self.step()

# -----------------------------------------------------------------------------
#  Multiple‑simulation helper 
# -----------------------------------------------------------------------------
 

    def run_simulation(self, ticks=250):
        self.setup()
        for _ in range(ticks):
            self.go()
        self.plot_opinion_sum_trend()
        self.plot_opinion_distribution()



def run_multiple_simulations(n_runs: int = 100, ticks: int = 250,
                            *, agent_count: int = 999, learning_rate: float = 0.3,
                            confidence_threshold: float = 0.5):
    all_sums = []
    for _ in range(n_runs):
        model = Model(agent_count=agent_count,
                      learning_rate=learning_rate,
                      confidence_threshold=confidence_threshold)
        model.run(ticks=ticks)
        all_sums.append(model.sum_opinion)

    all_sums = np.array(all_sums)
    mean = np.mean(all_sums, axis=0)
    std = np.std(all_sums, axis=0)
    ci = 1.96 * std / np.sqrt(n_runs)

    t = np.arange(ticks + 1)
    plt.figure(figsize=(3.5, 4))
    plt.plot(t, mean, label=f'Average over {n_runs} runs')
    plt.fill_between(t, mean - ci, mean + ci, alpha=0.1, label='95% CI')
    plt.xlabel('Time',fontsize=12)
    plt.ylabel('Sum of Expressed Opinions',fontsize=12)
    plt.text(-0.2, 1.1, '(a)', transform=plt.gca().transAxes,
         fontsize=15, fontweight='bold', va='top', ha='left')
    plt.xticks(np.arange(0, ticks+1, 50))  
    plt.yticks(np.arange(0, 75, 20)) 
    plt.xlim(left=0,right = ticks+1)
    plt.ylim(bottom=0,top=75)
    plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    plt.legend(loc='upper right', frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()



def create_panel_figure(n_runs=5, base_ticks=600):
    """Create a 4x3 panel figure with different parameter combinations."""

    learning_rates = [0.15, 0.30, 0.45]          # 行方向（μ）
    confidence_thresholds = [0.25, 0.50, 0.75, 1.00]  # 列方向（d）

    # d の値に応じて ticks を変更（今は全部同じでもOK）
    ticks_per_ct = {
        0.25: 450,
        0.50: 450,
        0.75: 450,
        1.00: 450
    }

    # Figure と GridSpec を作成
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 4, figure=fig, wspace=0.2, hspace=0.2)

    # あとで列タイトルを付けるために axes を 2D 配列で保持
    axes = [[None for _ in confidence_thresholds] for _ in learning_rates]

    # ---- 1st pass: シミュレーションを回して統計量を保存 ----
    all_results = {}

    print("Running simulations...")
    for i, lr in enumerate(learning_rates):
        for j, ct in enumerate(confidence_thresholds):
            ticks = ticks_per_ct[ct]
            print(f"Running μ={lr}, CT={ct}, Ticks={ticks}...")

            all_sums = []
            for run in range(n_runs):
                print(f"  Run {run+1}/{n_runs}")
                model = Model(agent_count=999,
                              learning_rate=lr,
                              confidence_threshold=ct)
                model.run(ticks=ticks)
                all_sums.append(model.sum_opinion)

            all_sums = np.array(all_sums)
            mean = np.mean(all_sums, axis=0)
            std = np.std(all_sums, axis=0)
            ci = 1.96 * std / np.sqrt(n_runs)

            all_results[(i, j)] = (mean, ci, ticks)

    # 全パネル共通の軸スケール
    unified_x_max = 450
    unified_x_tick_step = 150
    # 列ごとの y 軸最大値（左から順に）
    unified_y_max_per_col = [150, 120, 90, 75]

    print("Creating plots...")
    # ---- 2nd pass: プロット作成 ----
    for i, lr in enumerate(learning_rates):
        for j, ct in enumerate(confidence_thresholds):
            ax = fig.add_subplot(gs[i, j])
            axes[i][j] = ax                      # axes 配列に保存

            mean, ci, ticks = all_results[(i, j)]

            # 時間軸を統一（足りない分は最後の値で埋める）
            t = np.arange(unified_x_max + 1)
            if len(mean) < len(t):
                extended_mean = np.concatenate(
                    [mean, np.full(len(t) - len(mean), mean[-1])]
                )
                extended_ci = np.concatenate(
                    [ci, np.full(len(t) - len(ci), ci[-1])]
                )
            else:
                extended_mean = mean[:len(t)]
                extended_ci = ci[:len(t)]

            ax.plot(t, extended_mean, color='#56B4E9', linewidth=2.0)
            ax.fill_between(t,
                            extended_mean - extended_ci,
                            extended_mean + extended_ci,
                            alpha=0.15, color='#56B4E9')

            # 軸設定（xは共通、yは列ごとに最大値を割り当て）
            ax.set_xlim(0, unified_x_max)
            y_max_col = unified_y_max_per_col[j]
            ax.set_ylim(0, y_max_col)
            ax.set_xticks(np.arange(0, unified_x_max + 1, unified_x_tick_step))
            # y軸は0〜y_max_colまで、目盛りを程よく分割（3分割目安）
            y_step = max(int(y_max_col // 3), 1)
            ax.set_yticks(np.arange(0, y_max_col + 1, y_step))

            # 下段だけ x ラベル
            if i == len(learning_rates) - 1:
                ax.set_xlabel('Time Step (t)', fontsize=14, fontweight='medium')

            # 左端だけ y ラベル
            if j == 0:
                ax.set_ylabel('Sum of Expressed Opinions',
                              fontsize=14, fontweight='medium')

            ax.grid(True, linestyle='-', linewidth=0.5,
                    color='gray', alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=9)

    # ---- 列タイトル（d = ...）を一番上に ----
    for j, ct in enumerate(confidence_thresholds):
        axes[0][j].set_title(f'd = {ct}',
                             fontsize=16, fontweight='bold', pad=8)

    # ---- 行タイトル（μ = ...）を図の左側に横書きで ----
    for i, lr in enumerate(learning_rates):
        # 各行の中央に配置するための正確な位置計算
        y_pos = 0.77 - i * 0.275  # 各行の中央位置
        # μラベルを横書きで左側に配置（十分な余白を確保）
        fig.text(0.01, y_pos, f'μ = {lr}', va='center', ha='left',
                 fontsize=16, fontweight='bold', rotation=0)

    # 余白調整（μラベル位置に合わせて左余白をさらに広げる）
    plt.tight_layout(rect=[0.2, 0.04, 0.98, 0.94])
    plt.savefig('parameter_comparison_grid_new.png',
                dpi=300, bbox_inches='tight')
    plt.savefig('parameter_comparison_grid_new.svg',
                dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    create_panel_figure(n_runs = 100)  # 基本のticksはすでに関数内で定義

