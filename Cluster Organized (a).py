import numpy as np
import matplotlib.pyplot as plt
import random


# -----------------------------------------------------------------------------
#  Agent definition
# ----------------------------------------------------------------------------/

random.seed(25)
np.random.seed(25)

class Turtle:
    """Single agent with an opinion and a knowledge flag."""


    def __init__(self, x: float | None = None, y: float | None = None, latent_opinion: float | None = None):
        self.x = random.random() if x is None else x
        self.y = random.random() if y is None else y
        self.latent_opinion = np.random.normal(0, 0.5) if latent_opinion is None else latent_opinion  # fixed enthusiasm
        self.expressed_opinion = None  # evolving, only set when aware
        self.know = 0  # 0 = ignorant, 1 = knowledgeable
# -----------------------------------------------------------------------------
#  Opinion‑dynamics model
# ----------------------------------------------------------------------------/


class Model:
    """Opinion dynamics with **no ignorant‑ignorant interactions**."""


    def __init__(self, agent_count: int = 999, *, learning_rate: float = 0.3, confidence_threshold: float = 0.5):
        self.agent_count = agent_count
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold


        # runtime containers
        self.turtles = []
        self.tick = 0
        self.know_count = []
        self.ticks = []
       
        # 初期意見と最終意見の記録用
        self.initial_opinions = []
        self.opinion_snapshots = {}
        self.knowledge_snapshots = {}


    def setup(self):
        self.turtles.clear()
        self.tick = 0
        self.know_count.clear()
        self.ticks.clear()
       
        # 初期意見と最終意見の記録をリセット
        self.initial_opinions = []
        self.opinion_snapshots = {}
        self.knowledge_snapshots = {}


        for _ in range(self.agent_count):
            self.turtles.append(Turtle())

        # Developer agent: always aware, latent_opinion=1, expressed_opinion=1
        developer = Turtle(latent_opinion=1.0)
        developer.expressed_opinion = 1.0
        developer.know = 1
        self.turtles.append(developer)

        # 初期意見（latent_opinion）を記録
        self.initial_opinions = [t.latent_opinion for t in self.turtles]


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
        know_agents = [t for t in self.turtles if t.know == 1]
        self.know_count.append(len(know_agents))
        self.ticks.append(self.tick)


    def run(self, ticks: int = 400, snapshot_times=[20, 50, 80, 100, 130]):
        self.setup()
        # 時間0の状態を記録
        self.opinion_snapshots[0] = [t.expressed_opinion if t.know == 1 else t.latent_opinion for t in self.turtles]
        self.knowledge_snapshots[0] = [t.know for t in self.turtles]
        for _ in range(ticks):
            self.step()
            # 指定されたタイムステップでスナップショットを保存
            if self.tick in snapshot_times:
                self.opinion_snapshots[self.tick] = [t.expressed_opinion if t.know == 1 else t.latent_opinion for t in self.turtles]
                self.knowledge_snapshots[self.tick] = [t.know for t in self.turtles]
   
    # 1次元クラスタリング用の関数
    def cluster_opinions_1d(self, opinions, d):
        """
        Parameters
        ----------
        opinions : array-like, shape (n_agents,)
            各エージェントの意見値（数値スカラー）
        d : float
            クラスタ分割のしきい値（塊と塊の「谷間」の幅）
        Returns
        -------
        labels : ndarray, shape (n_agents,)
            各エージェントに対応するクラスタ ID（0, 1, 2, ...）
        clusters : list[list[int]]
            オリジナルのインデックスを保持したクラスタ構造
        """
        opinions = np.asarray(opinions)
        # 元のインデックスを保持しつつソート
        order = np.argsort(opinions)
        sorted_vals = opinions[order]


        # 隣接差分を取り，d 以上ならそこがクラスター境界
        boundaries = np.where(np.diff(sorted_vals) >= d)[0]


        # 連番のクラスタ ID を作成
        labels_sorted = np.zeros_like(sorted_vals, dtype=int)
        for cid, start in enumerate(np.r_[0, boundaries + 1]):
            end = boundaries[cid] + 1 if cid < len(boundaries) else len(sorted_vals)
            labels_sorted[start:end] = cid


        # 元の順序に戻す
        labels = np.empty_like(labels_sorted)
        labels[order] = labels_sorted


        # クラスタ構造（元インデックスのリスト）も返すと便利
        clusters = [
            order[labels_sorted == cid].tolist() for cid in range(labels_sorted.max() + 1)
        ]
        return labels, clusters


    def plot_opinion_evolution_clusters_1d_horizontal(self, time_points=[15, 25, 40, 75, 120]):
        """指定された時点での意見クラスタの進化を可視化（1次元クラスタリング手法を使用）- 横並びパネル版"""
        # 必要なスナップショットがあるか確認
        missing_points = [t for t in time_points if t not in self.opinion_snapshots]
        if missing_points:
            print(f"警告: 次の時点のデータがありません: {missing_points}")
            time_points = [t for t in time_points if t in self.opinion_snapshots]


        if not time_points:
            print("可視化する時点がありません。")
            return
       
        # サブプロットの作成 - 横方向に配置
        fig, axes = plt.subplots(1, len(time_points), figsize=(4*len(time_points), 5), sharey=True)
       
        # 1つの時点だけの場合、axesはスカラーになるので調整
        if len(time_points) == 1:
            axes = [axes]


        # 固定された色マッピング
        cluster_colors = {
            'positive': '#1f77b4',  # 青
            'center': '#2ca02c',    # 緑
            'negative': '#ffcc00'   # 黄色
        }
       
        # 各時点でのクラスタを描画
        for idx, tick in enumerate(time_points):
            ax = axes[idx]
           
            # その時点での意見と知識状態を取得
            opinions = self.opinion_snapshots[tick]
            knowledge = self.knowledge_snapshots[tick]
           
            # 認知者のデータのみを抽出
            aware_indices = [i for i, k in enumerate(knowledge) if k == 1]
           
            if len(aware_indices) <= 1:
                ax.text(0.5, 0.5, f"Time {tick}\nNot enough aware agents",
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                continue
           
            # 認知者の初期意見とその時点の意見
            initial_opinions = [self.initial_opinions[i] for i in aware_indices]
            current_opinions = [opinions[i] for i in aware_indices]
           
            # 1次元クラスタリングを実行（現在の意見でクラスタを決定）
            labels, clusters = self.cluster_opinions_1d(current_opinions, self.confidence_threshold)
            n_clusters = len(clusters)
           
            # クラスター情報を出力
            print(f"\n=== Time {tick} ===")
            print(f"Total aware agents: {len(aware_indices)}")
            print(f"Number of clusters: {n_clusters}")
           
            # 各クラスタの平均意見を計算して色を割り当て
            cluster_mean_opinions = []
            for i in range(n_clusters):
                cluster_indices = clusters[i]
                cluster_mean = np.mean([current_opinions[j] for j in cluster_indices])
                cluster_mean_opinions.append(cluster_mean)
               
                # クラスター詳細情報を出力
                cluster_size = len(cluster_indices)
                print(f"  Cluster {i+1}: {cluster_size} agents (mean opinion: {cluster_mean:.3f})")
           
            # 散布図 - クラスターごとに描画
            for i in range(n_clusters):
                cluster_indices = clusters[i]
                cluster_points = np.array([[current_opinions[j], initial_opinions[j]] for j in range(len(initial_opinions)) if labels[j] == i])
               
                # クラスタの平均意見に基づいて色を決定
                mean_opinion = cluster_mean_opinions[i]
                if mean_opinion > 0.3:
                    color = cluster_colors['positive']
                    cluster_label = f'Positive Cluster'
                elif mean_opinion < -0.3:
                    color = cluster_colors['negative']
                    cluster_label = f'Negative Cluster'
                else:
                    color = cluster_colors['center']
                    cluster_label = f'Central Cluster'
               
                if len(cluster_points) > 0:
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                              c=color, label=cluster_label if idx == len(time_points) - 1 else "",
                              s=60, alpha=0.7, edgecolors='w',zorder =2)
       
            # 各クラスタの平均意見を星印で表示
            for i in range(n_clusters):
                cluster_indices = [j for j in range(len(labels)) if labels[j] == i]
                if cluster_indices:
                    mean_initial = np.mean([initial_opinions[j] for j in cluster_indices])
                    mean_current = np.mean([current_opinions[j] for j in cluster_indices])
                   
                    # クラスタの平均意見に基づいて星印の色を決定
                    if mean_current > 0.3:
                        star_color = 'darkblue'
                    elif mean_current < -0.3:
                        star_color = 'darkorange'
                    else:
                        star_color = 'darkgreen'
                   
                    ax.scatter([mean_current], [mean_initial], c=star_color, s=150, alpha=0.8, marker='*', zorder=3)
                   
                    # 数値表示を見やすく改善
                    ax.text(mean_current + 0.1, mean_initial, f'{mean_current:.2f}',
                           fontsize=14, va='center', ha='left',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                           zorder=4, color='black')
       
            # 認知率を計算
            awareness_rate = sum(knowledge) / len(knowledge) * 100
           
            # 未認知者の処理
            unaware_indices = [i for i, k in enumerate(knowledge) if k == 0]
            if unaware_indices:
                unaware_initial = [self.initial_opinions[i] for i in unaware_indices]
               
                # 未認知者数も出力
                print(f"  Unaware agents: {len(unaware_indices)}")
               
                # 100%の抽出率で代表点をプロット
                sampling_rate = 1.0
                if len(unaware_initial) > 0:
                    sample_size = max(1, int(len(unaware_initial) * sampling_rate))
                    sampled_indices = random.sample(range(len(unaware_initial)), sample_size)
                    display_dots = [unaware_initial[i] for i in sampled_indices]
                   
                    ax.scatter(display_dots, display_dots,
                               c='darkgray', s=20, alpha=0.4, edgecolors='none', zorder=1)
       
            # グラフの設定 - パネルラベルとタイトルを統合
            panel_labels = ['(a)', '(b)', '(c)', '(d)']
            if idx < len(panel_labels):
                ax.set_title(f'{panel_labels[idx]} t = {tick}', fontsize=16, loc='left', fontweight='bold')
            else:
                ax.set_title(f't = {tick}', fontsize=16, loc='left')
            ax.grid(True, alpha=0.2)
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
           
            # X軸ラベル（時刻ごとの意見）
            ax.set_xlabel(f'Expressed Opinion', fontsize=16)
           
            # Y軸ラベル（初期意見）は最初のサブプロットにのみ表示
            if idx == 0:
                ax.set_ylabel('Latent Opinion', fontsize=16)
           
            # クラスター数と認知率を表示
            ax.text(0.05, 0.95, f'Clusters : {n_clusters}\nAware: {awareness_rate:.1f}%',
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(facecolor='white', alpha=0.7), va='top')


        # プロット設定
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, wspace=0.05)
        plt.savefig('opinion_clusters_horizontal2.png', dpi=300, bbox_inches='tight')
        plt.savefig('opinion_clusters_horizontal2.svg')
        plt.show()




if __name__ == '__main__':
    # パラメータの設定
    TICKS = 250
    AGENT_COUNT = 999
   
    # 単一シミュレーション実行
    print("Running simulation...")
    model = Model(agent_count=AGENT_COUNT, learning_rate=0.3, confidence_threshold=0.5)
    model.run(ticks=TICKS, snapshot_times=[15,30,75,200])
   
    # 横並びパネルでの可視化
    print("Plotting horizontal panel visualization...")
    model.plot_opinion_evolution_clusters_1d_horizontal([15,30,75,200])