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

    def __init__(self, x: float | None = None, y: float | None = None, latent_opinion: float | None = None):
        self.x = random.random() if x is None else x
        self.y = random.random() if y is None else y
        self.latent_opinion = np.random.normal(0, 0.5) if latent_opinion is None else latent_opinion
        self.expressed_opinion = None
        self.know = 0  # 0 = ignorant, 1 = knowledgeable


class MediaAgent:
    """Base media agent class."""
    
    def __init__(self, media_type="mean"):
        self.opinion = 0.0
        self.media_type = media_type  # "mean", "strong_media"
    
    def update_opinion(self, turtles):
        """Update media agent's opinion based on media type."""
        aware_agents = [t for t in turtles if t.know == 1 and t.expressed_opinion is not None]
        
        if not aware_agents:
            self.opinion = 0.0
            return
            
        if self.media_type == "mean":
            # Weak Media: 平均的な意見を出す
            self.opinion = sum(t.expressed_opinion for t in aware_agents) / len(aware_agents)
        elif self.media_type == "strong_media":
            # Strong Media: 出力は平均的意見（ただし影響力は強く、認知者にも作用する）
            self.opinion = sum(t.expressed_opinion for t in aware_agents) / len(aware_agents)


# -----------------------------------------------------------------------------
#  Opinion‑dynamics model
# -----------------------------------------------------------------------------

class Model:
    """Opinion dynamics with different media patterns."""

    def __init__(self, agent_count: int = 999, *, learning_rate: float = 0.3, 
                 confidence_threshold: float = 0.5, media_probability: float = 0.05,
                 media_type: str = "mean"):
        self.agent_count = agent_count
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.media_probability = media_probability
        self.media_type = media_type

        # runtime containers
        self.turtles = []
        # No Media の場合はMediaAgentを作成しない
        self.media_agent = None if media_type == "no_media" else MediaAgent(media_type)
        self.tick = 0
        self.sum_opinion = []
        self.know_count = []
        self.opinion_distribution = []
        self.ticks = []
        self.aware_positive_count = []
        self.media_stopped_tick = None
        self.new_aware_count = 0  # 新しく認知した人の数（各ステップでリセット）
    # coverage_count removed: we no longer record per-step media coverage

    def setup(self):
        self.turtles.clear()
        self.tick = 0
        self.sum_opinion.clear()
        self.know_count.clear()
        self.opinion_distribution.clear()
        self.ticks.clear()
        self.aware_positive_count.clear()
        self.media_stopped_tick = None
        self.new_aware_count = 0
    # coverage tracking removed

        for _ in range(self.agent_count):
            self.turtles.append(Turtle())

        # Developer agent: always aware, latent_opinion=1, expressed_opinion=1
        developer = Turtle(latent_opinion=1.0)
        developer.expressed_opinion = 1.0
        developer.know = 1
        self.turtles.append(developer)
        
        # Initialize media agent (no_mediaの場合は作らない)
        if self.media_type != "no_media":
            self.media_agent = MediaAgent(self.media_type)
            self.media_agent.update_opinion(self.turtles)

        self._record()

    def step(self):
        # No Media の場合はメディア関連の処理をスキップ
        if self.media_agent is not None:
            # Step 1: Update media agent opinion based on media type
            self.media_agent.update_opinion(self.turtles)
            
            # Step 2: Media agent interactions (no stopping at 100% awareness)
            awareness_rate = len([t for t in self.turtles if t.know == 1]) / len(self.turtles)
            
            # Media probability logic:
            # Both weak (mean) and strong media use the same coverage probability
            current_media_prob = self.media_probability
            
            # Reset new aware count AFTER using it for coverage calculation
            self.new_aware_count = 0
            
            # We no longer separately record per-step coverage probability
                
            if current_media_prob > 0:
                for agent in self.turtles:
                    if random.random() < current_media_prob:
                        self._interact_with_media(agent)
        
        # Step 3: Regular agent interactions
        knowledgeable_agents = [t for t in self.turtles if t.know == 1]
        random.shuffle(knowledgeable_agents)

        for a in knowledgeable_agents:
            candidates = [p for p in self.turtles if p is not a]
            b = random.choice(candidates)

            if b.know == 0:
                if abs(a.expressed_opinion - b.latent_opinion) < self.confidence_threshold:
                    b.expressed_opinion = (1 - self.learning_rate) * b.latent_opinion + self.learning_rate * a.expressed_opinion
                    b.know = 1
                    self.new_aware_count += 1  # 新しく認知した人をカウント
            elif b.know == 1:
                if abs(a.expressed_opinion - b.expressed_opinion) < self.confidence_threshold:
                    new_a = (1 - self.learning_rate) * a.expressed_opinion + self.learning_rate * b.expressed_opinion
                    new_b = (1 - self.learning_rate) * b.expressed_opinion + self.learning_rate * a.expressed_opinion
                    a.expressed_opinion, b.expressed_opinion = new_a, new_b

        self.tick += 1
        self._record()

    def _interact_with_media(self, agent):
        """Handle interaction between an agent and the media agent."""
        if self.media_type == "strong_media":
            # Strong media: 未認知者は認知しつつメディア意見に引き寄せられ、
            # 認知者もメディアによって説得されうる
            if agent.know == 0:
                agent.expressed_opinion = (1 - self.learning_rate) * agent.latent_opinion + self.learning_rate * self.media_agent.opinion
                agent.know = 1
                self.new_aware_count += 1
            else:
                # 認知者に対してもメディアが影響する（信頼範囲を考慮）
                if agent.expressed_opinion is not None and abs(agent.expressed_opinion - self.media_agent.opinion) < self.confidence_threshold:
                    agent.expressed_opinion = (1 - self.learning_rate) * agent.expressed_opinion + self.learning_rate * self.media_agent.opinion
        else:
            # Mean (弱い) media: 未認知者のみ説得可能（従来のFixed Coverage挙動）
            if agent.know == 0:
                agent.expressed_opinion = (1 - self.learning_rate) * agent.latent_opinion + self.learning_rate * self.media_agent.opinion
                agent.know = 1
                self.new_aware_count += 1

    def _record(self):
        know_agents = [t for t in self.turtles if t.know == 1]
        self.sum_opinion.append(sum(t.expressed_opinion for t in know_agents if t.expressed_opinion is not None))
        self.know_count.append(len(know_agents))
        aware_positive_agents = [t for t in know_agents if t.expressed_opinion is not None and t.expressed_opinion > 0.5]
        self.aware_positive_count.append(len(aware_positive_agents))
        self.opinion_distribution.append([t.expressed_opinion for t in know_agents if t.expressed_opinion is not None])
        self.ticks.append(self.tick)
        # coverage recording removed

    def run(self, ticks: int = 400):
        self.setup()
        for _ in range(ticks):
            self.step()


def run_three_media_comparison(n_runs: int = 100, ticks: int = 250,
                              *, agent_count: int = 999, learning_rate: float = 0.3,
                              confidence_threshold: float = 0.5, media_probability: float = 0.01):
    """3つのメディアパターンを2×3グリッドで比較するシミュレーション"""
    
    # 比較：No Media vs Weak Media vs Strong Media
    media_types = ["no_media", "mean", "strong_media"]
    media_labels = ["No Media", "Weak Media", "Strong Media"]
    
    results = {}
    
    for media_type in media_types:
        print(f"\nRunning simulations for {media_type} media...")
        all_aware_positive_counts = []
        all_sum_opinions = []
        
        for i in range(n_runs):
            print(f"Simulation {i+1}/{n_runs}...", end="", flush=True)
            
            # 各メディアタイプで同じシード値を使用して公平な比較を行う
            random.seed(42 + i)
            np.random.seed(42 + i)
            
            model = Model(agent_count=agent_count,
                         learning_rate=learning_rate,
                         confidence_threshold=confidence_threshold,
                         media_probability=media_probability,
                         media_type=media_type)
            model.run(ticks=ticks)
            all_aware_positive_counts.append(model.aware_positive_count)
            all_sum_opinions.append(model.sum_opinion)
            print(" done")
        
        # 平均値を計算
        all_aware_positive_counts = np.array(all_aware_positive_counts)
        all_sum_opinions = np.array(all_sum_opinions)
        
        results[media_type] = {
            'aware_positive': np.mean(all_aware_positive_counts, axis=0),
            'sum_opinion': np.mean(all_sum_opinions, axis=0),
            'std_aware_positive': np.std(all_aware_positive_counts, axis=0),
            'std_sum_opinion': np.std(all_sum_opinions, axis=0)
        }
    
    # プロット作成：2行×3列（Sum of Opinions, Number of Advocates）
    t = np.arange(ticks + 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 行ごとの色設定（青→緑）
    row_colors = ['#1f77b4', '#2ca02c']  # 青、緑

    # 行のラベル
    row_labels = ['Sum of Opinions', 'Number of Advocates']

    for col, media_type in enumerate(media_types):
        label = media_labels[col]

        # 第1行: Sum of Opinions (青)
        color = row_colors[0]
        mean_data = results[media_type]['sum_opinion']
        std_data = results[media_type]['std_sum_opinion']
        ci_data = 1.96 * std_data / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_data)

        axes[0, col].plot(t, mean_data, color=color, linewidth=2)
        axes[0, col].fill_between(t, mean_data - ci_data, mean_data + ci_data, alpha=0.1, color=color)
        axes[0, col].set_title(f'{label}', fontsize=12, fontweight='bold')
        axes[0, col].set_xlim(0, ticks)
        # 1行目のy軸範囲を統一
        axes[0, col].set_ylim(0, 80)
        # 目盛り: 4目盛り程度（5点）
        axes[0, col].set_yticks(np.linspace(0, 80, 5))
        axes[0, col].set_xticks(np.linspace(0, ticks, 5))
        axes[0, col].grid(True, alpha=0.3)
        if col == 0:
            axes[0, col].set_ylabel('Sum of Expressed Opinions', fontsize=12)

        # 第2行: Number of Advocates (緑)
        color = row_colors[1]
        mean_data = results[media_type]['aware_positive']
        std_data = results[media_type]['std_aware_positive']
        ci_data = 1.96 * std_data / np.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_data)

        axes[1, col].plot(t, mean_data, color=color, linewidth=2)
        axes[1, col].fill_between(t, mean_data - ci_data, mean_data + ci_data, alpha=0.1, color=color)
        axes[1, col].set_xlim(0, ticks)
        axes[1, col].set_ylim(0, 80)
        # 目盛り: 4目盛り程度（5点）
        axes[1, col].set_yticks(np.linspace(0, 80, 5))
        axes[1, col].set_xticks(np.linspace(0, ticks, 5))
        axes[1, col].grid(True, alpha=0.3)
        if col == 0:
            axes[1, col].set_ylabel('Number of Advocates', fontsize=12)
        axes[1, col].set_xlabel('Time Step (t)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('Three_Media_Comparison_2x3.png', dpi=300, bbox_inches='tight')
    plt.savefig('Three_Media_Comparison_2x3.svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 結果の数値サマリーを表示
    print("\n" + "="*60)
    print("SIMULATION RESULTS SUMMARY")
    print("="*60)
    print(f"Parameters: {n_runs} runs, {ticks} ticks, {agent_count} agents")
    print(f"Learning rate: {learning_rate}, Confidence threshold: {confidence_threshold}")
    print(f"Media probability: {media_probability}")
    print("-"*60)
    
    for i, media_type in enumerate(media_types):
        final_advocates = results[media_type]['aware_positive'][-1]
        final_opinion_sum = results[media_type]['sum_opinion'][-1]
        print(f"{media_labels[i]:>15}: {final_advocates:6.1f} advocates, {final_opinion_sum:6.1f} opinion sum")
    
    return results


if __name__ == '__main__':
    print("Comparing two media patterns in 2x2 grid: Weak vs Strong")
    print("1. No Media: メディアが存在しないベースライン")
    print("2. Weak Media: 固定の報道確率で未認知者のみ説得")
    print("3. Strong Media: 固定の報道確率で、認知者にも影響を与えるメディア")
    
    results = run_three_media_comparison(
        n_runs = 100, 
        ticks=200, 
        agent_count=999, 
        media_probability=0.01
    )