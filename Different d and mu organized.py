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
    
    learning_rates = [0.15, 0.30, 0.45]
    confidence_thresholds = [0.25, 0.50, 0.75, 1.00]
    
    # dの値に応じてticksを変更
    ticks_per_ct = {
        0.25: 450,
        0.50: 450,
        0.75: 450,
        1.00: 450
    }
    
    # Create a square figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig)  
    
    # First pass: Run simulations and collect statistics
    all_results = {}
    panel_max_values = {}  # Track max values for each individual panel
    
    print("Running simulations...")
    for i, lr in enumerate(learning_rates):
        for j, ct in enumerate(confidence_thresholds):
            # dの値に応じてticksを設定
            ticks = ticks_per_ct[ct]
            
            print(f"Running μ={lr}, CT={ct}, Ticks={ticks}...")
            # Run multiple simulations for this parameter combination
            all_sums = []
            for run in range(n_runs):
                print(f"  Run {run+1}/{n_runs}")
                model = Model(agent_count=999, 
                              learning_rate=lr, 
                              confidence_threshold=ct)
                model.run(ticks=ticks)
                all_sums.append(model.sum_opinion)
            
            # Calculate statistics
            all_sums = np.array(all_sums)
            mean = np.mean(all_sums, axis=0)
            std = np.std(all_sums, axis=0)
            ci = 1.96 * std / np.sqrt(n_runs)
            
            all_results[(i, j)] = (mean, ci, ticks)  # ticksも保存
            
            # Store panel-specific max value
            panel_max = np.max(mean + ci)
            panel_max_values[(i, j)] = panel_max
    
    # Set unified parameters for all panels
    unified_x_max = 450  # x軸の最大値を450に設定
    unified_x_tick_step = 150  # x軸の目盛り間隔（0, 150, 300, 450）
    unified_y_max = 150  # y軸の最大値を150に固定
    unified_y_tick_step = 50  # y軸の目盛り間隔（0, 50, 100, 150）
    
    print("Creating plots...")
    # Second pass: Create plots with row-unified y-limits
    for i, lr in enumerate(learning_rates):
        for j, ct in enumerate(confidence_thresholds):
            # Create subplot
            ax = fig.add_subplot(gs[i, j])
            
            mean, ci, ticks = all_results[(i, j)]  # ticksも取得
            
            # Plot on current subplot - use unified x-axis range
            t = np.arange(unified_x_max + 1)
            # Extend or truncate data to match unified x-axis
            if len(mean) < len(t):
                # Extend with last value if data is shorter
                extended_mean = np.concatenate([mean, np.full(len(t) - len(mean), mean[-1])])
                extended_ci = np.concatenate([ci, np.full(len(t) - len(ci), ci[-1])])
            else:
                # Truncate if data is longer
                extended_mean = mean[:len(t)]
                extended_ci = ci[:len(t)]
            
            ax.plot(t, extended_mean, color='blue', linewidth=1.5)
            ax.fill_between(t, extended_mean - extended_ci, extended_mean + extended_ci, alpha=0.1, color='blue')
            
            # Set subplot title and labels
            # Convert i,j coordinates to a letter: (0,0)=a, (0,1)=b, etc.
            label_idx = i * len(confidence_thresholds) + j
            label = chr(97 + label_idx)  # 97 is ASCII for 'a'
            
            # タイトルにパネル番号を含める
            ax.set_title(f'({label}) μ={lr}, d={ct}', fontsize=15, fontweight='bold', ha='left', x=0.0)
            
            # Use unified axis limits and tick marks for all panels
            ax.set_xlim(0, unified_x_max)
            ax.set_ylim(0, unified_y_max)
            ax.set_xticks(np.arange(0, unified_x_max+1, unified_x_tick_step))
            ax.set_yticks(np.arange(0, unified_y_max+1, unified_y_tick_step))
            
            # Add x-label for bottom row
            if i == 2:
                ax.set_xlabel('Time Step (t)', fontsize=14, fontweight='medium')
            
            # Add y-label for leftmost column
            if j == 0:
                ax.set_ylabel('Sum of Expressed Opinions', fontsize=14, fontweight='medium')
                
            # Add grid but make it lighter
            ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.2)
            
            # Adjust tick label size
            ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Add parameter labels to the figure
    fig.text(0.5, 0.01, 'Confidence Threshold Values (d)', ha='center', fontweight='bold', fontsize=18)
    fig.text(0.01, 0.5, 'Learning Rate Values (μ)', va='center', rotation='vertical', fontweight='bold', fontsize=18)
    
    # 余白を増やして外側のラベルのためのスペースを確保
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    plt.savefig('parameter_comparison_grid_variable_ticks.png', dpi=300, bbox_inches='tight')
    plt.savefig('parameter_comparison_grid_variable_ticks.svg', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    create_panel_figure(n_runs=100)  # 基本のticksはすでに関数内で定義
