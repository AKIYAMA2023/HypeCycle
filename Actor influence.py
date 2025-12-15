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

    def __init__(self, x: float | None = None, y: float | None = None, latent_opinion: float | None = None, is_stubborn: bool = False):
        self.x = random.random() if x is None else x
        self.y = random.random() if y is None else y
        self.latent_opinion = np.random.normal(0, 0.5) if latent_opinion is None else latent_opinion  # fixed enthusiasm
        self.expressed_opinion = None  # evolving, only set when aware
        self.know = 0  # 0 = ignorant, 1 = knowledgeable
        self.is_stubborn = is_stubborn  # True = stubborn, cannot be persuaded by others


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

    def __init__(self, agent_count: int = 999, *, learning_rate: float = 0.3, confidence_threshold: float = 0.5, stubborn_advocates: int = 0, stubborn_opponents: int = 0):
        self.agent_count = agent_count
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.stubborn_advocates = stubborn_advocates
        self.stubborn_opponents = stubborn_opponents

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

        # Create base population of ignorant agents
        for _ in range(self.agent_count):
            self.turtles.append(Turtle())

        # Developer agent: always aware, latent_opinion=1, expressed_opinion=1
        developer = Turtle(latent_opinion=1.0)
        developer.expressed_opinion = 1.0
        developer.know = 1
        self.turtles.append(developer)

        # Add stubborn advocate agents as external actors: initially ignorant but with strong positive latent opinion
        # These are additional agents beyond the base population
        # They start ignorant (know=0) but once they become aware, they cannot be persuaded by others
        for _ in range(self.stubborn_advocates):
            # Position stubborn advocates in the range (0.5-1.5, 0.5-1.5)
            x_pos = np.random.uniform(1.0, 1.0)
            y_pos = np.random.uniform(1.0, 1.0)
            stubborn = Turtle(x=x_pos, y=y_pos, latent_opinion=np.random.uniform(0.9, 1.0), is_stubborn=True)
            # stubborn agents start ignorant like others, but have strong positive latent opinions
            stubborn.know = 0  # Start ignorant
            stubborn.expressed_opinion = None  # No expressed opinion initially
            self.turtles.append(stubborn)

        # Add stubborn opponent agents as external actors: initially ignorant but with strong negative latent opinion
        # These are additional agents beyond the base population
        # They start ignorant (know=0) but once they become aware, they cannot be persuaded by others
        for _ in range(self.stubborn_opponents):
            # Position stubborn opponents in the range (-0.5--1.5, -0.5--1.5)
            x_pos = np.random.uniform(-1.0, -1.5)
            y_pos = np.random.uniform(-1.0, -1.5)
            stubborn_opp = Turtle(x=x_pos, y=y_pos, latent_opinion=np.random.uniform(-1.0, -0.9), is_stubborn=True)
            # stubborn opponents start ignorant like others, but have strong negative latent opinions
            stubborn_opp.know = 0  # Start ignorant
            stubborn_opp.expressed_opinion = None  # No expressed opinion initially
            self.turtles.append(stubborn_opp)

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
                    # Update only non-stubborn agents
                    new_a = a.expressed_opinion
                    new_b = b.expressed_opinion
                    
                    if not a.is_stubborn:
                        new_a = (1 - self.learning_rate) * a.expressed_opinion + self.learning_rate * b.expressed_opinion
                    if not b.is_stubborn:
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

# 複数シミュレーションの実行と平均化
def run_multiple_simulations_aware_positive(n_runs: int = 100, ticks: int = 250,
                                          *, agent_count: int = 999, learning_rate: float = 0.3,
                                          confidence_threshold: float = 0.5, stubborn_advocates: int = 0, stubborn_opponents: int = 0):
    all_aware_positive_counts = []
    
    print(f"Running {n_runs} simulations, {ticks} ticks each...")
    
    for i in range(n_runs):
        print(f"Simulation {i+1}/{n_runs}...", end="", flush=True)
        model = Model(agent_count=agent_count,
                      learning_rate=learning_rate,
                      confidence_threshold=confidence_threshold,
                      stubborn_advocates=stubborn_advocates,
                      stubborn_opponents=stubborn_opponents)
        model.run(ticks=ticks)
        all_aware_positive_counts.append(model.aware_positive_count)
        print(" done")

    print("All simulations completed. Creating plot...")
    
    # 平均と信頼区間の計算
    all_aware_positive_counts = np.array(all_aware_positive_counts)
    
    mean_aware_positive = np.mean(all_aware_positive_counts, axis=0)
    std_aware_positive = np.std(all_aware_positive_counts, axis=0)
    ci_aware_positive = 1.96 * std_aware_positive / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_aware_positive)
    
    # プロット
    t = np.arange(ticks + 1)
    
    plt.figure(figsize=(3.5, 4))
    
    plt.plot(t, mean_aware_positive, color='green', label=f'Average over {n_runs} runs')
    if n_runs > 1:
        plt.fill_between(t, mean_aware_positive - ci_aware_positive, mean_aware_positive + ci_aware_positive, 
                        alpha=0.1, color='green', label='95% CI')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Number of Advocates', fontsize=12)
    #plt.title('Evolution of Number of Proponents', fontsize=14)
    plt.xlim(left=0,right = ticks+1)
    plt.ylim(bottom=0,top=120)
    plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    plt.xticks(np.arange(0, ticks+1, 50))  
    plt.yticks(np.arange(0, 130, 20)) 
    #plt.legend(loc='none', fontsize=9,frameon=False)
    plt.text(-0.2, 1.1, '(b)', transform=plt.gca().transAxes,
         fontsize=15, fontweight='bold', va='top', ha='left')
    plt.tight_layout()
    plt.savefig('aware_positive_opinion_trend1.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mean_aware_positive

# 新しい関数：awareなエージェントの意見の総和とadvocateの数を並べてプロット
def run_multiple_simulations_combined_plots(n_runs: int = 100, ticks: int = 250,
                                          *, agent_count: int = 999, learning_rate: float = 0.3,
                                          confidence_threshold: float = 0.5, stubborn_advocates: int = 0, stubborn_opponents: int = 0):
    all_sum_opinions = []
    all_aware_positive_counts = []
    
    print(f"Running {n_runs} simulations, {ticks} ticks each...")
    
    for i in range(n_runs):
        print(f"Simulation {i+1}/{n_runs}...", end="", flush=True)
        model = Model(agent_count=agent_count,
                      learning_rate=learning_rate,
                      confidence_threshold=confidence_threshold,
                      stubborn_advocates=stubborn_advocates,
                      stubborn_opponents=stubborn_opponents)
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
    ax1.set_ylim(bottom=0, top=120)
    ax1.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    ax1.set_xticks(np.arange(0, ticks+1, 50))
    ax1.set_yticks(np.arange(0, 130, 20))
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
    ax2.set_ylim(bottom=0, top=120)
    ax2.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    ax2.set_xticks(np.arange(0, ticks+1, 50))
    ax2.set_yticks(np.arange(0, 130, 20))
    ax2.text(-0.22, 1.1, '(b)', transform=ax2.transAxes,
             fontsize=15, fontweight='bold', va='top', ha='left')
    
    plt.tight_layout()
    plt.savefig('Actor Influence.png', dpi=300, bbox_inches='tight')
    plt.savefig ('Actor Influence.svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mean_sum_opinion, mean_aware_positive

def run_stubborn_comparison_plots(n_runs: int = 50, ticks: int = 200,
                                *, agent_count: int = 999, learning_rate: float = 0.3,
                                confidence_threshold: float = 0.5, 
                                stubborn_advocates_count: int = 5,
                                stubborn_opponents_count: int = 5):
    """
    4つの条件で比較：
    1. Baseline (No stubborn agents)
    2. Only stubborn advocates  
    3. Only stubborn opponents
    4. Both stubborn advocates and opponents
    
    2×4のレイアウト:
    1行目: Sum of opinions for each condition
    2行目: Number of advocates for each condition
    """
    
    conditions = [
        {"name": "Baseline", "advocates": 0, "opponents": 0},
        {"name": "Postitive Opinion Leaders", "advocates": stubborn_advocates_count, "opponents": 0},
        {"name": "Negative Opinion Leaders", "advocates": 0, "opponents": stubborn_opponents_count},
        {"name": "Both Opinion Leaders", "advocates": stubborn_advocates_count, "opponents": stubborn_opponents_count}
    ]
    
    results = []
    
    for i, condition in enumerate(conditions):
        print(f"Running condition {i+1}/4: {condition['name']} ({condition['advocates']} advocates, {condition['opponents']} opponents)...")
        
        all_sum_opinions = []
        all_aware_positive_counts = []
        
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}...", end="", flush=True)
            model = Model(agent_count=agent_count,
                         learning_rate=learning_rate,
                         confidence_threshold=confidence_threshold,
                         stubborn_advocates=condition['advocates'],
                         stubborn_opponents=condition['opponents'])
            model.run(ticks=ticks)
            all_sum_opinions.append(model.sum_opinion)
            all_aware_positive_counts.append(model.aware_positive_count)
            print(" done")
        
        # 平均計算
        mean_sum_opinion = np.mean(all_sum_opinions, axis=0)
        mean_aware_positive = np.mean(all_aware_positive_counts, axis=0)
        std_sum_opinion = np.std(all_sum_opinions, axis=0)
        std_aware_positive = np.std(all_aware_positive_counts, axis=0)
        ci_sum_opinion = 1.96 * std_sum_opinion / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_sum_opinion)
        ci_aware_positive = 1.96 * std_aware_positive / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_aware_positive)
        
        results.append({
            'condition': condition,
            'mean_sum_opinion': mean_sum_opinion,
            'mean_aware_positive': mean_aware_positive,
            'ci_sum_opinion': ci_sum_opinion,
            'ci_aware_positive': ci_aware_positive
        })
    
    # 2×4のサブプロット作成
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    t = np.arange(ticks + 1)
    
    # 1行目: Sum of Opinions for each condition
    for col, result in enumerate(results):
        ax = axes[0, col]
        
        # Plot sum of opinions (blue)
        ax.plot(t, result['mean_sum_opinion'], color='blue', linewidth=2)
        if n_runs > 1:
            ax.fill_between(t, 
                           result['mean_sum_opinion'] - result['ci_sum_opinion'],
                           result['mean_sum_opinion'] + result['ci_sum_opinion'],
                           alpha=0.1, color='blue')
        
        # Formatting
        ax.set_xlim(0, ticks)
        ax.set_ylim(-5, 85)
        ax.set_yticks(np.arange(-5, 86, 25))
        ax.set_title(result['condition']['name'], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Remove x-axis labels and ticks for top row but keep grid lines
        ax.set_xticks(np.arange(0, ticks+1, 50))
        
        if col == 0:
            ax.set_ylabel('Sum of Expressed Opinions', fontsize=14)
    
    # 2行目: Number of Advocates for each condition  
    for col, result in enumerate(results):
        ax = axes[1, col]
        
        # Plot number of advocates (green)
        ax.plot(t, result['mean_aware_positive'], color='green', linewidth=2)
        if n_runs > 1:
            ax.fill_between(t,
                            result['mean_aware_positive'] - result['ci_aware_positive'],
                            result['mean_aware_positive'] + result['ci_aware_positive'],
                            alpha=0.1, color='green')
        
        # Formatting
        ax.set_xlim(0, ticks)
        ax.set_ylim(0, 70)
        ax.set_yticks(np.arange(0, 71, 20))
        ax.set_xlabel('Time Step (t)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        if col == 0:
            ax.set_ylabel('Number of Advocates', fontsize=14)
    
    plt.tight_layout()
    
    # Adjust spacing to reduce gaps between plots
    plt.subplots_adjust(hspace=0.15, wspace=0.2)
    plt.savefig('stubborn_agents_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('stubborn_agents_comparison.svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == '__main__':
    # 単一シミュレーション実行例
    #print("Running single simulation...")
    #model = Model(agent_count=999, learning_rate=0.3, confidence_threshold=0.5)
    #model.run(ticks=250)
    #model.plot_aware_positive_trend()
    
    # 複数シミュレーション実行例（従来の単一図）
    #print("\nRunning multiple simulations...")
    #run_multiple_simulations_aware_positive(n_runs=100, ticks=200, agent_count=999)
    
    # 新しい並列図の実行例
    #print("\nRunning multiple simulations for combined plots...")
    #run_multiple_simulations_combined_plots(n_runs=100, ticks=200, agent_count=999)
    
    # 頑固な肯定派を含むシミュレーション例
    #print("\nRunning simulations with stubborn advocates...")
    #run_multiple_simulations_combined_plots(n_runs=10, ticks=200, agent_count=999, stubborn_advocates=5)
    
    # 頑固エージェントの比較プロット
    print("\nRunning stubborn agents comparison...")
    run_stubborn_comparison_plots(n_runs=100, ticks=250, agent_count=999, 
                                stubborn_advocates_count=10, stubborn_opponents_count=10)