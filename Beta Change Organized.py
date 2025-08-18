import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import gennorm
from scipy.special import gamma

# -----------------------------------------------------------------------------
#  Agent definition
# -----------------------------------------------------------------------------

class Turtle:
    """Single agent with an opinion and a knowledge flag."""

    def __init__(self, latent_opinion: float, x: float | None = None, y: float | None = None):
        self.x = random.random() if x is None else x
        self.y = random.random() if y is None else y
        self.latent_opinion = latent_opinion
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
    """

    def __init__(self, agent_count: int = 999, *, learning_rate: float = 0.3, confidence_threshold: float = 0.5,
                 opinion_loc: float = 0.0, opinion_scale: float = 0.5, opinion_beta: float = 2.0):
        self.agent_count = agent_count
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.opinion_loc = opinion_loc
        self.opinion_scale = opinion_scale
        self.opinion_beta = opinion_beta 

        self.turtles = []
        self.tick = 0
        self.sum_opinion = []
        self.know_count = []
        self.opinion_distribution = []
        self.ticks = []
        self.initial_opinions_all = [] # To store all initial opinions
        self.high_opinion_aware_agents_count = [] # New metric
        self.high_opinion_threshold = 0.0 # To store Mean + 2SD threshold
        self.positive_opinion_aware_agents_count = [] # Track aware agents with opinion > 0.5
        self.one_sigma_aware_agents_count = [] # Track aware agents with opinion >= μ+1σ
        self.sum_expressed_opinions = [] # Track sum of all aware agents' opinions

    def setup(self):
        self.turtles.clear()
        self.tick = 0
        self.sum_opinion.clear()
        self.know_count.clear()
        self.opinion_distribution.clear()
        self.ticks.clear()
        self.initial_opinions_all.clear() 
        self.high_opinion_aware_agents_count.clear()
        self.positive_opinion_aware_agents_count.clear()
        self.one_sigma_aware_agents_count.clear()
        self.sum_expressed_opinions.clear()

        if self.opinion_beta <= 0:
            raise ValueError("opinion_beta must be greater than 0 for gennorm and gamma function.")

        std_dev_initial_dist = self.opinion_scale * np.sqrt(gamma(3.0/self.opinion_beta) / gamma(1.0/self.opinion_beta))
        self.high_opinion_threshold = self.opinion_loc + 2 * std_dev_initial_dist
        self.one_sigma_threshold = self.opinion_loc + 1 * std_dev_initial_dist

        for _ in range(self.agent_count):
            if self.opinion_beta == 2.0:
                latent_op = np.random.normal(self.opinion_loc, 0.5)
            else:
                latent_op = gennorm.rvs(beta=self.opinion_beta, loc=self.opinion_loc, scale=self.opinion_scale)
            t = Turtle(latent_opinion=latent_op)
            self.turtles.append(t)
            self.initial_opinions_all.append(latent_op)

        # Developer agent: always aware, expressed_opinion=1.0
        developer_latent = 1.0
        developer = Turtle(latent_opinion=developer_latent)
        developer.expressed_opinion = 1.0
        developer.know = 1
        self.turtles.append(developer)
        self.initial_opinions_all.append(developer_latent)

        self._record()

    def step(self):
        knowledgeable_agents = [t for t in self.turtles if t.know == 1]
        random.shuffle(knowledgeable_agents)

        for a in knowledgeable_agents:
            candidates = [p for p in self.turtles if p is not a]
            if not candidates: continue
            b = random.choice(candidates)

            # Use expressed_opinion for aware agents, latent_opinion for ignorant
            a_op = a.expressed_opinion if a.know == 1 else a.latent_opinion
            b_op = b.expressed_opinion if b.know == 1 else b.latent_opinion
            interact = (abs(a_op - b_op) < self.confidence_threshold)
            if not interact:
                continue

            if a.know == 1 and b.know == 0:
                # b becomes aware, sets expressed_opinion
                b.expressed_opinion = b.latent_opinion + self.learning_rate * (a.expressed_opinion - b.latent_opinion)
                b.know = 1
            elif a.know == 1 and b.know == 1:
                new_a = a.expressed_opinion + self.learning_rate * (b.expressed_opinion - a.expressed_opinion)
                new_b = b.expressed_opinion + self.learning_rate * (a.expressed_opinion - b.expressed_opinion)
                a.expressed_opinion, b.expressed_opinion = new_a, new_b

        self.tick += 1
        self._record()

    def _record(self):
        know_agents = [t for t in self.turtles if t.know == 1]
        self.sum_opinion.append(sum(t.expressed_opinion for t in know_agents if t.expressed_opinion is not None))
        self.know_count.append(len(know_agents))

        # Count aware agents with expressed_opinion >= threshold
        count_high_opinion_aware = sum(1 for t in know_agents if t.expressed_opinion is not None and t.expressed_opinion >= self.high_opinion_threshold)
        self.high_opinion_aware_agents_count.append(count_high_opinion_aware)

        # Count aware agents with expressed_opinion > 0.5
        count_positive_opinion_aware = sum(1 for t in know_agents if t.expressed_opinion is not None and t.expressed_opinion > 0.5)
        self.positive_opinion_aware_agents_count.append(count_positive_opinion_aware)

        # Count aware agents with expressed_opinion >= μ+1σ
        count_one_sigma_aware = sum(1 for t in know_agents if t.expressed_opinion is not None and t.expressed_opinion >= self.one_sigma_threshold)
        self.one_sigma_aware_agents_count.append(count_one_sigma_aware)

        # Sum of all expressed opinions (aware agents' opinions)
        sum_expressed = sum(t.expressed_opinion for t in know_agents if t.expressed_opinion is not None)
        self.sum_expressed_opinions.append(sum_expressed)

        self.ticks.append(self.tick)

    def run(self, ticks: int = 400):
        self.setup()
        for _ in range(ticks):
            self.step()

# -----------------------------------------------------------------------------
#  Multiple‑simulation helper 
# -----------------------------------------------------------------------------
def run_multiple_simulations(n_runs: int = 3, ticks: int = 250,
                            *, agent_count: int = 999, learning_rate: float = 0.3,
                            confidence_threshold: float = 0.5,
                            opinion_loc: float = 0.0, opinion_scale: float = 0.5,
                            opinion_beta: float = 2.0,
                            ax_for_plot=None, ax_for_sum_plot=None, ax_for_one_sigma_plot=None): 
    all_high_opinion_counts = [] 
    all_positive_opinion_counts = []
    all_one_sigma_counts = []
    all_sum_expressed_opinions = []
    initial_total_agents_above_threshold_at_t0 = 0 # To store the count at t=0

    for i in range(n_runs):
        model = Model(agent_count=agent_count,
                      learning_rate=learning_rate,
                      confidence_threshold=confidence_threshold,
                      opinion_loc=opinion_loc,
                      opinion_scale=opinion_scale,
                      opinion_beta=opinion_beta)
        model.run(ticks=ticks) # setup() is called within run()
        
        all_high_opinion_counts.append(model.high_opinion_aware_agents_count) 
        all_positive_opinion_counts.append(model.positive_opinion_aware_agents_count) 
        all_one_sigma_counts.append(model.one_sigma_aware_agents_count)
        all_sum_expressed_opinions.append(model.sum_expressed_opinions) 
        
        if i == 0:
            # Calculate the total number of agents (aware or not) whose initial opinion
            # was >= high_opinion_threshold (μ_initial + 2*SD_initial) at t=0
            # This uses the state from the very first model instance after its setup.
            threshold_at_t0 = model.high_opinion_threshold 
            count_at_t0 = sum(1 for opinion_val in model.initial_opinions_all if opinion_val >= threshold_at_t0)
            initial_total_agents_above_threshold_at_t0 = count_at_t0

    all_high_opinion_counts = np.array(all_high_opinion_counts)
    mean_high_count = np.mean(all_high_opinion_counts, axis=0)
    std_high_count = np.std(all_high_opinion_counts, axis=0)
    
    all_positive_opinion_counts = np.array(all_positive_opinion_counts)
    mean_positive_count = np.mean(all_positive_opinion_counts, axis=0)
    std_positive_count = np.std(all_positive_opinion_counts, axis=0)
    
    all_one_sigma_counts = np.array(all_one_sigma_counts)
    mean_one_sigma_count = np.mean(all_one_sigma_counts, axis=0)
    std_one_sigma_count = np.std(all_one_sigma_counts, axis=0)
    
    all_sum_expressed_opinions = np.array(all_sum_expressed_opinions)
    mean_sum_expressed = np.mean(all_sum_expressed_opinions, axis=0)
    std_sum_expressed = np.std(all_sum_expressed_opinions, axis=0)
    
    if n_runs > 1:
        ci_high_count = 1.96 * std_high_count / np.sqrt(n_runs)
        ci_positive_count = 1.96 * std_positive_count / np.sqrt(n_runs)
        ci_one_sigma_count = 1.96 * std_one_sigma_count / np.sqrt(n_runs)
        ci_sum_expressed = 1.96 * std_sum_expressed / np.sqrt(n_runs)
    else:
        ci_high_count = np.zeros_like(mean_high_count)
        ci_positive_count = np.zeros_like(mean_positive_count)
        ci_one_sigma_count = np.zeros_like(mean_one_sigma_count)
        ci_sum_expressed = np.zeros_like(mean_sum_expressed)

    t = np.arange(ticks + 1)

    # ax_for_plot is not used anymore - first row shows only PDF

    if ax_for_sum_plot:
        ax_for_sum_plot.plot(t, mean_sum_expressed, label='Sum of Expressed Opinions', color='blue')
        if n_runs > 1:
            ax_for_sum_plot.fill_between(t, mean_sum_expressed - ci_sum_expressed, mean_sum_expressed + ci_sum_expressed, alpha=0.1, color='blue')
        ax_for_sum_plot.set_xticks(np.arange(0, ticks + 1, max(1, ticks // 4)))
        ax_for_sum_plot.set_xlim(left=0, right=ticks)
        ax_for_sum_plot.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
        ax_for_sum_plot.tick_params(axis='both', which='major', labelsize=8)

    if ax_for_one_sigma_plot:
        ax_for_one_sigma_plot.plot(t, mean_one_sigma_count, label='Avg. Aware (Op ≥ μ+1σ)', color='green')
        if n_runs > 1:
            ax_for_one_sigma_plot.fill_between(t, mean_one_sigma_count - ci_one_sigma_count, mean_one_sigma_count + ci_one_sigma_count, alpha=0.1, color='green')
        ax_for_one_sigma_plot.set_xlabel('Time', fontsize=10)
        ax_for_one_sigma_plot.set_xticks(np.arange(0, ticks + 1, max(1, ticks // 4)))
        ax_for_one_sigma_plot.set_xlim(left=0, right=ticks)
        ax_for_one_sigma_plot.set_ylim(bottom=0, top=75)  # Expand y-axis range to match Simulation 100.py 
        ax_for_one_sigma_plot.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
        ax_for_one_sigma_plot.tick_params(axis='both', which='major', labelsize=8)

    else: 
        plt.figure(figsize=(3.5, 4))
        plt.plot(t, mean_high_count, label=f'Aware (Op ≥ μ+2σ) ({n_runs} runs)', color='blue')
        plt.plot(t, mean_positive_count, label=f'Aware (Op > 0.5) ({n_runs} runs)', color='red')
        if n_runs > 1:
            plt.fill_between(t, mean_high_count - ci_high_count, mean_high_count + ci_high_count, alpha=0.1, color='blue')
            plt.fill_between(t, mean_positive_count - ci_positive_count, mean_positive_count + ci_positive_count, alpha=0.1, color='red')
        plt.xlabel('Time',fontsize=12)
        plt.ylabel('Number of Aware Agents',fontsize=12)
        plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
        plt.legend(loc='upper right', frameon=False, fontsize=9)
        plt.tight_layout()
        plt.show()
    
    return initial_total_agents_above_threshold_at_t0 # Return the calculated initial count

if __name__ == '__main__':
    beta_values_to_plot = [1.0, 1.5, 2.0, 3.0, 4.0] 
    n_sim_runs_for_avg = 100  
    simulation_total_ticks = 300 

    # Calculate figure size: each panel 3.5x4, 5 columns x 3 rows
    panel_width = 3.5
    panel_height = 4.0
    total_width = panel_width * len(beta_values_to_plot) + 0.5  # 3.5 * 5 = 17.5
    total_height = panel_height * 3 + 0.5 # 4.0 * 3 = 12.0
    
    fig, axes = plt.subplots(3, len(beta_values_to_plot), figsize=(total_width, total_height), sharey='row')

    # β=2.0, scale=0.5, loc=0でSD=0.5になることを基準とする
    reference_beta = 2.0
    reference_scale = 0.5
    target_sd = 0.5  # 全β値で統一したい標準偏差

    # 各β値で目標標準偏差を達成するためのscaleを事前計算
    beta_to_scale = {}
    
    # 基準値（β=2.0）の確認
    reference_scale_factor = np.sqrt(gamma(3.0/reference_beta) / gamma(1.0/reference_beta))
    actual_sd_for_reference = reference_scale * reference_scale_factor
    print(f"基準値確認: β={reference_beta}, scale={reference_scale} → SD={actual_sd_for_reference:.4f}")
    
    for beta_val in beta_values_to_plot:
        if beta_val == 2.0:
            # β=2.0の場合は、Simulation 100.pyと同じパラメータを使用
            # 正規分布では直接np.random.normal(0, 0.5)を使うが、
            # gennormでの計算用にscale=0.7071相当を設定
            required_scale = 0.7071  # This gives std=0.5 for β=2.0
            beta_to_scale[beta_val] = required_scale
            print(f"β={beta_val:.1f}: scale={required_scale:.4f} → SD=0.5 (normal distribution equivalent)")
        else:
            # 他のβ値では、β=2.0と同じ標準偏差になるようにscaleを調整
            scale_factor = np.sqrt(gamma(3.0/beta_val) / gamma(1.0/beta_val))
            required_scale = target_sd / scale_factor
            beta_to_scale[beta_val] = required_scale
            
            # 実際のSDを計算して確認
            actual_sd = required_scale * scale_factor
            print(f"β={beta_val:.1f}: scale={required_scale:.4f} → SD={actual_sd:.4f}")

    common_params = {
        'agent_count': 999,  # Same as Simulation 100.py
        'learning_rate': 0.3,
        'confidence_threshold': 0.5,
        'opinion_loc': 0.0,
        # opinion_scaleは各βで個別に設定するため、ここでは削除
    }

    print(f"\n全β値で標準偏差を{target_sd}に統一します")

    for col_idx, beta_val in enumerate(beta_values_to_plot):
        print(f"\n--- Processing for beta = {beta_val:.1f} ---")

        # Get the opinion_scale for this specific beta value
        current_opinion_scale = beta_to_scale[beta_val]

        initial_count_t0 = run_multiple_simulations( 
            n_runs=n_sim_runs_for_avg,
            ticks=simulation_total_ticks,
            opinion_beta=beta_val,
            opinion_scale=current_opinion_scale,  # Pass the specific scale for this beta
            **common_params,
            ax_for_sum_plot=axes[1, col_idx],
            ax_for_one_sigma_plot=axes[2, col_idx]
        )

        ax_hist = axes[0, col_idx]
        
        current_std_dev = current_opinion_scale * np.sqrt(gamma(3.0/beta_val) / gamma(1.0/beta_val))
        current_threshold = common_params['opinion_loc'] + 2 * current_std_dev

        # 統一されたx軸範囲を使用（すべてのβ値で同じ）
        hist_min = -2  # 固定値
        hist_max = 2   # 固定値
        
        x_pdf = np.linspace(hist_min, hist_max, 200)
        pdf_values = gennorm.pdf(x_pdf, beta_val, loc=common_params['opinion_loc'], scale=current_opinion_scale)
        ax_hist.plot(x_pdf, pdf_values, 'gray', lw=2, alpha=0.8, label='Theoretical PDF')
        
        # Remove the dashed line for μ+2σ
        # ax_hist.axvline(current_threshold, color='black', linestyle='--', lw=1.5, label=f'μ+2σ ({current_threshold:.2f})')
        
        ax_hist.set_title(f'β={beta_val:.1f}', fontsize=15)
        ax_hist.set_xlabel('Latent Opinion', fontsize=12)
        if col_idx == 0:
            ax_hist.set_ylabel('Density', fontsize=12)
        ax_hist.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
        ax_hist.set_xlim(hist_min, hist_max)
        ax_hist.tick_params(axis='both', which='major', labelsize=9)
        
        # Add panel label (a), (b), (c), (d), (e) for first row - outside panel, top-left
        panel_label = chr(ord('a') + col_idx)
        ax_hist.text(-0.06, 1.01, f'({panel_label})', transform=ax_hist.transAxes, 
                    fontsize=18, fontweight='bold', va='bottom', ha='center')

        # --- Plotting for the second row (sum of expressed opinions) ---
        ax_sum = axes[1, col_idx]
        if col_idx == 0:
            ax_sum.set_ylabel('Sum of Expressed Opinions', fontsize=12)
        
        # Add panel label (f), (g), (h), (i), (j) for second row - outside panel, top-left
        panel_label_sum = chr(ord('f') + col_idx)
        ax_sum.text(-0.06, 1.01, f'({panel_label_sum})', transform=ax_sum.transAxes, 
                   fontsize=18, fontweight='bold', va='bottom', ha='center')

        # --- Plotting for the third row (1σ opinion) ---
        ax_positive = axes[2, col_idx]
        ax_positive.set_xlabel('Time Step (t)', fontsize=12)
        if col_idx == 0:
            ax_positive.set_ylabel('Number of Advocates', fontsize=12)
        
        # Add panel label (k), (l), (m), (n), (o) for third row - outside panel, top-left
        panel_label_positive = chr(ord('k') + col_idx)
        ax_positive.text(-0.06, 1.01, f'({panel_label_positive})', transform=ax_positive.transAxes, 
                        fontsize=18, fontweight='bold', va='bottom', ha='center')


    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.8)  # より小さなパディング
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.97, hspace=0.2, wspace=0.15)  # より密集
    plt.savefig('3x5_with_two_metrics.png', dpi=300) # Updated filename
    plt.savefig('3x5_with_two_metrics.svg', format='svg') # SVG format
    plt.show()

