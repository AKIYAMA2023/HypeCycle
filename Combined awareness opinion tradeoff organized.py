import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

# -----------------------------------------------------------------------------
#  Agent definition
# -----------------------------------------------------------------------------

class Turtle:
    """Single agent with an opinion, latent opinion, and a knowledge flag."""

    def __init__(self, x: float | None = None, y: float | None = None, latent_opinion: float | None = None):
        self.x = random.random() if x is None else x
        self.y = random.random() if y is None else y
        self.latent_opinion = np.random.normal(0, 0.5) if latent_opinion is None else latent_opinion
        self.expressed_opinion = None  # only set when aware
        self.initial_opinion = self.latent_opinion  # 初期意見を保存
        self.know = 0  # 0 = ignorant, 1 = knowledgeable

# -----------------------------------------------------------------------------
#  Opinion‑dynamics model
# -----------------------------------------------------------------------------

class Model:
    """Opinion dynamics with **no ignorant‑ignorant interactions**."""

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

        # Arrays for opinion group metrics
        self.medium_positive_ratio = []  # Potential moderate advocates who became aware
        self.strong_positive_ratio = []  # Potential extreme advocates who became aware
        self.neutral_ratio = []          # Potential neutral agents who became aware
        self.moderate_opponents_ratio = []  # Potential moderate opponents who became aware
        self.extreme_opponents_ratio = []   # Potential extreme opponents who became aware

        # For enthusiasm tracking
        self.awareness_timing = {}  # key: agent id, value: tick when agent became aware
        self.initial_opinions = {}  # key: agent id, value: initial opinion
        self.recently_aware_initial_opinions = []  # 直近でawareになったエージェントの初期意見の平均の推移

    def setup(self):
        self.turtles.clear()
        self.tick = 0
        self.sum_opinion.clear()
        self.know_count.clear()
        self.opinion_distribution.clear()
        self.ticks.clear()
        self.medium_positive_ratio.clear()
        self.strong_positive_ratio.clear()
        self.neutral_ratio.clear()
        self.moderate_opponents_ratio.clear()
        self.extreme_opponents_ratio.clear()
        self.awareness_timing = {}
        self.initial_opinions = {}
        self.recently_aware_initial_opinions = []

        for i in range(self.agent_count):
            turtle = Turtle()
            self.turtles.append(turtle)
            self.initial_opinions[i] = turtle.latent_opinion

        # Developer agent: always aware, latent_opinion=1, expressed_opinion=1
        developer = Turtle(latent_opinion=1.0)
        developer.expressed_opinion = 1.0
        developer.initial_opinion = 1.0
        developer.know = 1
        self.turtles.append(developer)
        # 初期開発者の情報を記録
        last_idx = len(self.turtles) - 1
        self.awareness_timing[last_idx] = 0
        self.initial_opinions[last_idx] = developer.latent_opinion

        self._record()

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
                    # エージェントがawareになったタイミングを記録
                    agent_idx = self.turtles.index(b)
                    self.awareness_timing[agent_idx] = self.tick
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

    def _record(self):
        know_agents = [t for t in self.turtles if t.know == 1]
        self.sum_opinion.append(sum(t.expressed_opinion for t in know_agents if t.expressed_opinion is not None))
        self.know_count.append(len(know_agents))
        self.opinion_distribution.append([t.expressed_opinion for t in know_agents if t.expressed_opinion is not None])
        self.ticks.append(self.tick)

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

        # 直近でawareになったエージェントの初期意見
        recent_window = 10
        recent_aware_indices = [i for i, t in enumerate(self.awareness_timing.values()) 
                              if self.tick - t <= recent_window]
        if recent_aware_indices:
            recent_aware_agents_idx = [i for i in self.awareness_timing.keys() 
                                    if self.tick - self.awareness_timing[i] <= recent_window]
            recent_init_opinions = [self.initial_opinions[i] for i in recent_aware_agents_idx]
            recent_mean_init_opinion = np.mean(recent_init_opinions) if recent_init_opinions else None
        else:
            recent_mean_init_opinion = None
        self.recently_aware_initial_opinions.append(recent_mean_init_opinion)

    def run(self, ticks: int = 400):
        self.setup()
        for _ in range(ticks):
            self.step()

# -----------------------------------------------------------------------------
#  Style settings for consistent plotting
# -----------------------------------------------------------------------------

STYLE = {
    # Advocates: 
    "ext_adv": dict(color="skyblue", lw=3.5, alpha=1.0, zorder=5),
    "mod_adv": dict(color="skyblue", lw=2.5, ls="--", alpha=0.8, zorder=3),

    # Neutral: 
    "neutral": dict(color="#A9B7C6", lw=2.5, ls=":", alpha=0.7, zorder=2),

    # Opponents: 
    "ext_opp": dict(color="orange", lw=3.5, alpha=0.9, zorder=4),
    "mod_opp": dict(color="orange", lw=2.5, ls="--", alpha=0.8, zorder=3),
}

# matplotlib設定
plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.figsize": (7, 6),
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# -----------------------------------------------------------------------------
#  Combined plotting function
# -----------------------------------------------------------------------------

def run_combined_simulations(n_runs: int = 100, ticks: int = 200,
                           *, agent_count: int = 999, learning_rate: float = 0.3,
                           confidence_threshold: float = 0.5):
    """Run simulations and create combined plots."""
    
    # Containers for opinion group awareness
    med_adv_all = np.zeros((n_runs, ticks + 1))
    ext_adv_all = np.zeros_like(med_adv_all)
    neutral_all = np.zeros_like(med_adv_all)
    mod_opp_all = np.zeros_like(med_adv_all)
    ext_opp_all = np.zeros_like(med_adv_all)
    
    # Containers for enthusiasm tracking
    all_awareness_ratios = []
    all_recent_mean_init_opinions = []

    print(f"Running {n_runs} simulations, {ticks} ticks each...")
    
    for run in range(n_runs):
        print(f"Running simulation {run + 1}/{n_runs}…", end="\r")
        
        # Set seed for reproducibility in each run
        random.seed(run)
        np.random.seed(run)
        
        model = Model(agent_count=agent_count,
                      learning_rate=learning_rate,
                      confidence_threshold=confidence_threshold)
        model.run(ticks=ticks)

        # Opinion group data
        med_adv_all[run] = model.medium_positive_ratio
        ext_adv_all[run] = model.strong_positive_ratio
        neutral_all[run] = model.neutral_ratio
        mod_opp_all[run] = model.moderate_opponents_ratio
        ext_opp_all[run] = model.extreme_opponents_ratio
        
        # Enthusiasm data
        awareness_ratio = [count / (agent_count + 1) for count in model.know_count]
        all_awareness_ratios.append(awareness_ratio)
        
        initial_opinions = []
        last_valid_init = 0
        for op in model.recently_aware_initial_opinions:
            if op is not None:
                last_valid_init = op
            initial_opinions.append(last_valid_init)
        all_recent_mean_init_opinions.append(initial_opinions)

    print("\nCalculating averages and creating combined plot...")
    
    # Calculate means for opinion groups
    med_adv_mean = med_adv_all.mean(axis=0)
    ext_adv_mean = ext_adv_all.mean(axis=0)
    neutral_mean = neutral_all.mean(axis=0)
    mod_opp_mean = mod_opp_all.mean(axis=0)
    ext_opp_mean = ext_opp_all.mean(axis=0)
      # Calculate means for enthusiasm
    all_awareness_ratios = np.array(all_awareness_ratios)
    all_recent_mean_init_opinions = np.array(all_recent_mean_init_opinions)
    mean_awareness = np.mean(all_awareness_ratios, axis=0)
    mean_recent_init_opinion = np.mean(all_recent_mean_init_opinions, axis=0)    # Create combined figure with custom width ratios
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], figure=fig, wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # -------------------- Left plot: Opinion Group Awareness --------------------
    t = np.arange(ticks + 1)
    
    ax1.plot(t, ext_adv_mean * 100, label="Potential Extreme\nAdvocates", **STYLE["ext_adv"])
    ax1.plot(t, med_adv_mean * 100, label="Potential Moderate\nAdvocates", **STYLE["mod_adv"])
    ax1.plot(t, neutral_mean * 100, label="Potential Neutral\nAgents", **STYLE["neutral"])
    ax1.plot(t, mod_opp_mean * 100, label="Potential Moderate\nOpponents", **STYLE["mod_opp"])
    ax1.plot(t, ext_opp_mean * 100, label="Potential Extreme\nOpponents", **STYLE["ext_opp"])

    ax1.set_xlabel("Time Step (t)", fontsize=14)
    ax1.set_ylabel("Awareness Rate (%)", fontsize=14)
    ax1.set_ylim(-2, 102)
    ax1.set_facecolor('white')
    ax1.grid(alpha=0.3, linewidth=0.5)
    
    for spine in ax1.spines.values():
        spine.set_linewidth(1.0)

    legend1 = ax1.legend(loc="lower right", frameon=False, fancybox=False, shadow=False, 
                        framealpha=0.95, edgecolor='none', fontsize=12)
    legend1.get_frame().set_facecolor('#FFFFFF')
    ax1.text(-0.17, 1.06, '(a)', transform=ax1.transAxes,
             fontsize=18, fontweight='bold', va='top', ha='left')
    ax1.set_xticks(np.arange(0, ticks+1, 50))
    ax1.set_yticks(np.arange(0, 101, 20))
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # -------------------- Right plot: Awareness vs Recent Opinion --------------------    # 認知度をプロット (左軸)
    color1 = 'black'
    ax2.set_ylabel('Awareness Rate (%)', fontsize=14)
    ax2.set_xlabel('Time Step (t)', fontsize=14)
    line2 = ax2.plot(t, mean_awareness * 100, color=color1, lw=2, label='Awareness', alpha=0.6)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.set_ylim(-60, 110)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])

    # 直近Aware初期意見をプロット (右軸)
    color2 = '#F5B700'
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('Latent Opinions of Recently Aware Agents', fontsize=14)
    line1 = ax2_twin.plot(t, mean_recent_init_opinion, color=color2, linestyle='--', lw=2.5,
                    label='Opinion')
    ax2_twin.tick_params(axis='both', labelsize=10)
    ax2_twin.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2_twin.set_ylim(-0.6, 1.1)
    ax2_twin.axhline(0, ls="--", lw=1.5, color='darkgray')
    
    # 交差点を検出してグレーアウト
    crossing_points = []
    for i in range(1, len(mean_recent_init_opinion)):
        if mean_recent_init_opinion[i-1] >= 0 and mean_recent_init_opinion[i] < 0:
            crossing_points.append(i)
        elif mean_recent_init_opinion[i-1] < 0 and mean_recent_init_opinion[i] >= 0:
            crossing_points.append(i)
      # 交差点が2つ以上ある場合、間の区間の背景を変更
    if len(crossing_points) >= 2:
        for i in range(0, len(crossing_points)-1, 2):
            if i+1 < len(crossing_points):
                start = crossing_points[i]
                end = crossing_points[i+1]
                ax2.axvspan(start, end, alpha=0.15, color='gray')    # 凡例の結合
    lines = line2 + line1
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='lower right', frameon=False, fontsize=11)
    
    # グリッドと装飾
    ax2.grid(True, axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    ax2.text(-0.27, 1.06, '(b)', transform=ax2.transAxes,
             fontsize=18, fontweight='bold', va='top', ha='left')
    
    # 全体の調整
    plt.tight_layout()
    plt.savefig('combined_awareness_opinion_dynamics.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('combined_awareness_opinion_dynamics.svg', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    # 分析情報の出力
    if crossing_points:
        print("\nOpinion crossing points analysis:")
        for i, point in enumerate(crossing_points):
            direction = "positive to negative" if i % 2 == 0 else "negative to positive"
            print(f"  Time {point}: {direction}")
            
        if len(crossing_points) >= 2:
            for i in range(0, len(crossing_points)-1, 2):
                if i+1 < len(crossing_points):
                    start = crossing_points[i]
                    end = crossing_points[i+1]
                    start_awareness = mean_awareness[start]
                    end_awareness = mean_awareness[end]
                    duration = end - start
                    print(f"\nNegative opinion region:")
                    print(f"  Duration: {duration} time steps")
                    print(f"  Awareness at start: {start_awareness:.2%}")
                    print(f"  Awareness at end: {end_awareness:.2%}")
                    print(f"  Awareness change: {end_awareness - start_awareness:.2%}")

if __name__ == "__main__":
    # Run combined simulations
    run_combined_simulations(n_runs=100, ticks=200, learning_rate=0.3, confidence_threshold=0.5)
