import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

# 正規分布のパラメータ
mean = 0
std = 0.5

# x軸の範囲を設定
x = np.linspace(-1.6, 1.6, 1000)
y = stats.norm.pdf(x, mean, std)

# 各グループの範囲を定義（洗練された色パレット）
groups = [
    {'name': 'Potential\nExtreme\nOpponents', 'range': (-1.6, -1.0), 'color': '#d62728', 'alpha': 0.7},
    {'name': 'Potential\nModerate\nOpponents', 'range': (-1.0, -0.5), 'color': '#ff9999', 'alpha': 0.6},
    {'name': 'Potential\nNeutrals', 'range': (-0.5, 0.5), 'color': '#e6e6e6', 'alpha': 0.6},
    {'name': 'Potential\nModerate\nAdvocates', 'range': (0.5, 1.0), 'color': '#87ceeb', 'alpha': 0.6},
    {'name': 'Potential\nExtreme\nAdvocates', 'range': (1.0, 1.6), 'color': '#2e8b57', 'alpha': 0.7}
]

# 図を作成（より美しいサイズ）
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')

# 各グループの領域を塗りつぶし（分布曲線の下）
for i, group in enumerate(groups):
    x_fill = x[(x >= group['range'][0]) & (x <= group['range'][1])]
    y_fill = stats.norm.pdf(x_fill, mean, std)
    ax.fill_between(x_fill, 0, y_fill, alpha=group['alpha'], color=group['color'],
                   edgecolor='white', linewidth=0.5)
    
    # グループの境界に縦線を追加
    if i > 0:  # 最初のグループ以外
        ax.axvline(x=group['range'][0], color='gray', linestyle='-', 
                  linewidth=1, alpha=0.5, zorder=1)

# 正規分布を描画（より美しく）
ax.plot(x, y, color='#2c3e50', linewidth=2, label='Normal Distribution (μ=0, σ=0.5)', zorder=2)

# 上部に5つの区切られたセクションを作成（枠を大きくして文字が入るように）
y_top = 1.05
y_bottom = 0.85
x_start = -1.6
x_end = 1.6

# 外枠は削除（色分けされているため不要）

# 各グループのセクションを作成
for i, group in enumerate(groups):
    x_left = group['range'][0]
    x_right = group['range'][1]
    
    # セクションの背景を対応する色で塗りつぶし
    section_rect = Rectangle((x_left, y_bottom), x_right-x_left, y_top-y_bottom,
                           facecolor=group['color'], alpha=0.3, zorder=1)
    ax.add_patch(section_rect)
    
    # 縦の仕切り線も削除（色分けで十分区別できる）
    
    # セクション内にラベルを配置
    x_center = (x_left + x_right) / 2
    y_center = (y_top + y_bottom) / 2
    
    # 改行を保持したラベル名
    label_name = group['name']
    
    ax.text(x_center, y_center, label_name, 
            ha='center', va='center', fontsize=18, 
            color='black', zorder=1)

# X = 1.0の点線＋developer's opinionのラベルを追加
x_dev = 1.0
y_dev = stats.norm.pdf(x_dev, mean, std)

# 垂直線を追加
ax.axvline(x=x_dev, color='darkred', linestyle='--', linewidth=1, alpha=0.8)

# ラベルを追加
ax.annotate("Developer", 
            xy=(x_dev+0.02, y_dev+0.02), xytext=(x_dev + 0.3, y_dev + 0.2),
            ha='center', va='center', fontsize=16, 
            color='darkred', weight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
          )

# 該当部分に星印(枠線はdark red)を追加
ax.plot(x_dev, y_dev, '*', color='yellow', markersize=30, 
            markeredgecolor='darkred', markeredgewidth=2,
        label="Developer's Opinion")

# 軸とラベルの設定（より美しく）
ax.set_xlabel('Latent Opinion (Li)', fontsize=20, weight='bold', color='#2c3e50')
ax.set_ylabel('Probability Density', fontsize=20, weight='bold', color='#2c3e50')
#ax.set_title('Agent Groups Based on Normal Distribution of Latent Opinions\n(μ=0, σ=0.5)', 
           #  fontsize=20, weight='bold', color='#2c3e50', pad=25)

# x軸の表示範囲を調整して余白を減らす
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-0.02, 1.05)  # 枠の最上部と一致させる

# 目盛りの間隔を設定
ax.set_xticks(np.arange(-1.5, 2, 0.5))
ax.set_yticks(np.arange(0, 0.9, 0.2))

# 目盛りラベルのスタイル改善
ax.tick_params(axis='both', which='major', labelsize=13, 
               colors='#2c3e50', width=1.5)
ax.tick_params(axis='both', which='minor', width=1)

# より美しいグリッド
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.8, color='#bdc3c7')
ax.set_axisbelow(True)

# レイアウトを調整
plt.tight_layout()

# 各グループの確率を計算して表示
print("各グループの確率:")
for group in groups:
    prob = stats.norm.cdf(group['range'][1], mean, std) - stats.norm.cdf(group['range'][0], mean, std)
    print(f"{group['name']}: {prob:.3f} ({prob*100:.1f}%)")

# 背景色を設定（より美しく）
ax.set_facecolor('white')
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)  # 右の境界線を表示
ax.spines['left'].set_color('#2c3e50')
ax.spines['bottom'].set_color('#2c3e50')
ax.spines['right'].set_color('#2c3e50')  # 右の境界線の色も設定
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)  # 右の境界線の太さも設定

# 図を保存（plt.show()の前に実行）
plt.savefig('agent_groups_normal_distribution.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('agent_groups_normal_distribution.svg', bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("図を 'agent_groups_normal_distribution.png' と 'agent_groups_normal_distribution.svg' として保存しました")

# 図を表示
plt.show()