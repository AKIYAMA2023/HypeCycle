import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Fracking Cost.csvからデータをインポート
df = pd.read_csv('Fracking Cost.csv')

# 2017年を基準年として設定
base_year = 2017
base_ppi = df[df['Year'] == base_year]['PPI'].values[0]
base_cost = df[df['Year'] == base_year]['Permian Midland'].values[0]

# PPIデフレータ調整後のコスト（実質価格）を計算
df['Real_Cost_PPI'] = df['Permian Midland'] * (base_ppi / df['PPI'])

# グラフの作成
plt.figure(figsize=(8,4))
# Nominal costを控えめに、PPI Adjusted costを目立たせる
plt.plot(df['Year'], df['Permian Midland'], marker='o', linewidth=2, markersize=4, 
         label='Nominal Fracking Cost', color='#2E86AB', alpha=0.6, linestyle='--')
plt.plot(df['Year'], df['Real_Cost_PPI'], marker='s', linewidth=3, markersize=6, 
         label='PPI-Adjusted Fracking Cost (Real Price)', color='#A23B72', alpha=1.0)

plt.title('Fracking Cost Trends (2017-2024)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Cost per Barrel ($)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(df['Year'])
plt.yticks(range(46, 70, 6))  # 46から6ごとの目盛り設定
plt.tight_layout()

# 画像として保存
plt.savefig('fracking_cost_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('fracking_cost_comparison.svg', dpi=300, bbox_inches='tight')
plt.show()


