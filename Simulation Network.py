import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx

# Set seed values for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
#  Agent definition
# -----------------------------------------------------------------------------

class Turtle:
    """Single agent with an opinion and a knowledge flag."""

    def __init__(self, x: float | None = None, y: float | None = None, agent_id: int | None = None):
        self.x = random.random() if x is None else x
        self.y = random.random() if y is None else y
        self.opinion = np.random.normal(0, 0.5)  # initial opinion
        self.know = 0  # 0 = ignorant, 1 = knowledgeable
        self.color = None
        self.agent_id = agent_id  # Network node ID

    def refresh_color(self):
        norm = (max(min(self.opinion, 1.0), -1.0) + 1) / 2
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
    ▸ Agents interact only with their network neighbors.
    """

    def __init__(self, agent_count: int = 999, *, learning_rate: float = 0.3, confidence_threshold: float = 0.5, 
                 network_type: str = 'small_world', **network_params):
        self.agent_count = agent_count
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.network_type = network_type
        self.network_params = network_params

        # runtime containers
        self.turtles = []
        self.tick = 0
        self.sum_opinion = []
        self.know_count = []
        self.opinion_distribution = []
        self.ticks = []
        self.aware_positive_count = []  # 新しい指標: Aware かつ opinion > 0.5 の数
        self.network = None  # NetworkX graph

    # -------------------------------- setup ---------------------------------
    def setup(self):
        self.turtles.clear()
        self.tick = 0
        self.sum_opinion.clear()
        self.know_count.clear()
        self.opinion_distribution.clear()
        self.ticks.clear()

        # Create agents with network IDs
        for i in range(self.agent_count):
            self.turtles.append(Turtle(agent_id=i))

        developer = Turtle(agent_id=self.agent_count)
        developer.opinion = 1.0
        developer.know = 1
        self.turtles.append(developer)

        # Create network
        self._create_network()

        self._record()

    def _create_network(self):
        """Create network based on network_type parameter"""
        total_nodes = self.agent_count + 1  # +1 for developer
        
        if self.network_type == 'small_world':
            # Watts-Strogatz small-world network
            k = self.network_params.get('k', 10)  # Each node connects to k nearest neighbors
            p = self.network_params.get('p', 0.1)  # Probability of rewiring
            self.network = nx.watts_strogatz_graph(total_nodes, k, p, seed=RANDOM_SEED)
            
        elif self.network_type == 'scale_free':
            # Barabási-Albert scale-free network
            m = self.network_params.get('m', 5)  # Number of edges to attach from new node
            self.network = nx.barabasi_albert_graph(total_nodes, m, seed=RANDOM_SEED)
            
        elif self.network_type == 'regular':
            # Regular ring lattice
            k = self.network_params.get('k', 10)
            self.network = nx.watts_strogatz_graph(total_nodes, k, 0, seed=RANDOM_SEED)  # p=0 for regular
            
        elif self.network_type == 'random':
            # Erdős–Rényi random graph
            p = self.network_params.get('p', 0.01)
            self.network = nx.erdos_renyi_graph(total_nodes, p, seed=RANDOM_SEED)
            
        elif self.network_type == 'complete':
            # Complete graph (all-to-all connections)
            self.network = nx.complete_graph(total_nodes)
            
        else:
            raise ValueError(f"Unknown network type: {self.network_type}")
        
        # Ensure the network is connected
        if not nx.is_connected(self.network):
            # Add edges to make it connected
            components = list(nx.connected_components(self.network))
            for i in range(len(components) - 1):
                node1 = next(iter(components[i]))
                node2 = next(iter(components[i + 1]))
                self.network.add_edge(node1, node2)

    # -------------------------------- step ----------------------------------
    def step(self):
        
        knowledgeable_agents = [t for t in self.turtles if t.know == 1]
        random.shuffle(knowledgeable_agents)

        for a in knowledgeable_agents:
            # Get network neighbors instead of all agents
            neighbor_ids = list(self.network.neighbors(a.agent_id))
            if not neighbor_ids:
                continue  # Skip if no neighbors
            
            # Convert neighbor IDs to turtle objects
            neighbors = [self.turtles[nid] for nid in neighbor_ids if nid < len(self.turtles)]
            if not neighbors:
                continue
            
            b = random.choice(neighbors)

            # skip if partner is ignorant and would lead to ignorant‑ignorant pair afterwards
            # (but here a is know=1, so always at least one knowledgeable)

            # interaction condition: bounded confidence + logarithmic activation
            interact = (abs(a.opinion - b.opinion) < self.confidence_threshold)
            if not interact:
                continue

            if a.know == 1 and b.know == 0:
                # only ignorant partner updates
                b.opinion += self.learning_rate * (a.opinion - b.opinion)
                b.know = 1
            elif a.know == 1 and b.know == 1:
                # both knowledgeable → mutual update
                new_a = a.opinion + self.learning_rate * (b.opinion - a.opinion)
                new_b = b.opinion + self.learning_rate * (a.opinion - b.opinion)
                a.opinion, b.opinion = new_a, new_b
            # no else: a is knowledgeable by definition

        self.tick += 1
        self._record()

    # ------------------------------- logging --------------------------------
    def _record(self):
        know_agents = [t for t in self.turtles if t.know == 1]
        self.sum_opinion.append(sum(t.opinion for t in know_agents if t.opinion is not None))
        self.know_count.append(len(know_agents))
        
        # 新しい指標: Aware かつ opinion > 0.5 のエージェント数
        aware_positive_agents = [t for t in know_agents if t.opinion > 0.5]
        self.aware_positive_count.append(len(aware_positive_agents))
        
        self.opinion_distribution.append([t.opinion for t in know_agents])
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

    def plot_network_and_opinions(self, figsize=(12, 5)):
        """Plot network structure and opinion distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Network visualization
        pos = nx.spring_layout(self.network, k=1, iterations=50, seed=RANDOM_SEED)
        
        # Color nodes by opinion
        node_colors = []
        for node_id in self.network.nodes():
            if node_id < len(self.turtles):
                turtle = self.turtles[node_id]
                if turtle.know == 1:
                    # Normalize opinion to [0, 1] for color mapping
                    norm_opinion = (max(min(turtle.opinion, 1.0), -1.0) + 1) / 2
                    node_colors.append(plt.cm.RdYlBu_r(norm_opinion))
                else:
                    node_colors.append('lightgray')  # Ignorant agents
            else:
                node_colors.append('lightgray')
        
        nx.draw(self.network, pos, ax=ax1, node_color=node_colors, 
                node_size=30, edge_color='gray', alpha=0.7, width=0.5)
        ax1.set_title(f'{self.network_type.replace("_", " ").title()} Network\n(Colored by Opinion)')
        
        # Opinion distribution histogram
        aware_opinions = [t.opinion for t in self.turtles if t.know == 1]
        if aware_opinions:
            # Dynamic bin calculation based on data range and count
            n_opinions = len(aware_opinions)
            if n_opinions < 10:
                bins = min(n_opinions, 5)
            else:
                opinion_range = max(aware_opinions) - min(aware_opinions)
                if opinion_range < 0.1:
                    bins = min(n_opinions, 5)  # Very narrow range
                else:
                    bins = min(20, max(5, n_opinions // 3))
            
            ax2.hist(aware_opinions, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(0.5, color='red', linestyle='--', label='Advocate threshold (0.5)')
            ax2.set_xlabel('Opinion')
            ax2.set_ylabel('Number of Aware Agents')
            ax2.set_title('Opinion Distribution (Aware Agents)')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'network_and_opinions_{self.network_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

# 複数シミュレーションの実行と平均化
def run_multiple_simulations_aware_positive(n_runs: int = 100, ticks: int = 250,
                                          *, agent_count: int = 999, learning_rate: float = 0.3,
                                          confidence_threshold: float = 0.5):
    all_aware_positive_counts = []
    
    print(f"Running {n_runs} simulations, {ticks} ticks each...")
    
    for i in range(n_runs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Simulation {i+1}/{n_runs}...")
        model = Model(agent_count=agent_count,
                      learning_rate=learning_rate,
                      confidence_threshold=confidence_threshold)
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
    plt.ylim(bottom=0,top=75)
    plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    plt.xticks(np.arange(0, ticks+1, 50))  
    plt.yticks(np.arange(0, 75, 20)) 
    #plt.legend(loc='none', fontsize=9,frameon=False)
    plt.text(-0.2, 1.1, '(b)', transform=plt.gca().transAxes,
         fontsize=15, fontweight='bold', va='top', ha='left')
    plt.tight_layout()
    plt.savefig('aware_positive_opinion_trend1.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mean_aware_positive

# ネットワーク構造比較関数
def compare_network_structures(n_runs: int = 50, ticks: int = 200, agent_count: int = 999):
    """Compare different network structures"""
    network_configs = {
        'Complete': {'network_type': 'complete'},
        'Small World': {'network_type': 'small_world', 'k': 6, 'p': 0.1},
        'Scale Free': {'network_type': 'scale_free', 'm': 3},
        'Regular': {'network_type': 'regular', 'k': 6},
        'Random': {'network_type': 'random', 'p': 0.01}
    }
    
    results = {}
    
    for name, config in network_configs.items():
        print(f"Running simulations for {name} network...")
        all_aware_positive = []
        
    for i in range(n_runs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Run {i+1}/{n_runs}...")
        model = Model(agent_count=agent_count, **config)
        model.run(ticks=ticks)
        all_aware_positive.append(model.aware_positive_count)
        
    results[name] = np.mean(all_aware_positive, axis=0)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    t = np.arange(ticks + 1)
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, (name, mean_values) in enumerate(results.items()):
        plt.plot(t, mean_values, color=colors[i % len(colors)], 
                label=name, linewidth=2)
    
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Number of Advocates', fontsize=12)
    plt.title('Network Structure Comparison: Evolution of Advocates', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('network_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

# 新しい関数：awareなエージェントの意見の総和とadvocateの数を並べてプロット
def run_multiple_simulations_combined_plots(n_runs: int = 100, ticks: int = 250,
                                          *, agent_count: int = 999, learning_rate: float = 0.3,
                                          confidence_threshold: float = 0.5, **network_params):
    all_sum_opinions = []
    all_aware_positive_counts = []
    
    print(f"Running {n_runs} simulations, {ticks} ticks each...")
    
    for i in range(n_runs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Simulation {i+1}/{n_runs}...")
        model = Model(agent_count=agent_count,
                      learning_rate=learning_rate,
                      confidence_threshold=confidence_threshold,
                      **network_params)
        model.run(ticks=ticks)
        all_sum_opinions.append(model.sum_opinion)
        all_aware_positive_counts.append(model.aware_positive_count)

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
    ax2.set_ylim(bottom=0, top=75)
    ax2.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    ax2.set_xticks(np.arange(0, ticks+1, 50))
    ax2.set_yticks(np.arange(0, 75, 20))
    ax2.text(-0.22, 1.1, '(b)', transform=ax2.transAxes,
             fontsize=15, fontweight='bold', va='top', ha='left')
    
    plt.tight_layout()
    plt.savefig('reproducing_hype_cycle.png', dpi=300, bbox_inches='tight')
    plt.savefig ('reproducing_hype_cycle.svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mean_sum_opinion, mean_aware_positive


if __name__ == '__main__':
    # メイン研究: ネットワーク構造の比較分析
    print("=== Network Structure Impact Analysis ===")
    print(f"Random seed: {RANDOM_SEED} (for reproducibility)")
    print("Comparing different network structures with full parameters...")
    #compare_network_structures(n_runs=100, ticks=250, agent_count=999)
    
    # 従来モデル（完全グラフ）との比較
    #print("\n=== Traditional Model Comparison ===")
    #print("Running complete network (equivalent to original random interaction)...")
    #traditional_results = run_multiple_simulations_combined_plots(
        #n_runs=100, ticks=250, agent_count=999, 
        #network_type='complete')
    
    # Scale-free ネットワーク分析
    print("\n=== Scale-Free Network Analysis ===")
    print("Running Scale-Free network simulation...")
    scale_free_results = run_multiple_simulations_combined_plots(
         n_runs=100, ticks=250, agent_count=999, 
         network_type='scale_free', m=10)
    
    # random ネットワーク分析
    print("\n=== Random Network Analysis ===")
    print("Running Random network simulation...")
    random_results = run_multiple_simulations_combined_plots(
         n_runs=100, ticks=250, agent_count=999, 
         network_type='random', p=0.02)

    # small-world ネットワーク分析
    print("\n=== Small-World Network Analysis ===")
    print("Running Small-World network simulation...")
    small_world_results = run_multiple_simulations_combined_plots(
        n_runs=100, ticks=250, agent_count=999, 
        network_type='small_world', k=20, p=0.1)
    
    print("\n=== Analysis Complete ===")
    print("All network structure analyses have been completed.")
    print("Generated files:")
    print("- network_comparison.png: Comparison across all network types")
    print("- network_and_opinions_small_world.png: Small world network visualization")
    print("- reproducing_hype_cycle.png: Traditional and network-based results")