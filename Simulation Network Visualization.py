# Load the provided simulation code and generate a 3×3 panel of "number of proponents over time"
# across Random, Scale-free, and Small-world networks, for average degrees 40, 20, 10.
# 
# Implementation notes:
# - To respect the UI rule of "no subplots", each mini-chart is created in its own figure,
#   saved as a PNG, and then a final grid image is composed from those PNGs using PIL.
# - We keep runs modest so it executes quickly while still averaging out noise.
# - Mapping of "average degree" to network params:
#     * Random (Erdős–Rényi): p ≈ avg_deg / (N-1)
#     * Scale-free (Barabási–Albert): avg_deg ≈ 2m  ⇒ m = max(1, round(avg_deg/2))
#     * Small-world (Watts–Strogatz): avg_deg = k (rounded to nearest even)
#
import importlib.util, types, os, math, numpy as np, matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
from PIL import Image, ImageDraw, ImageFont

# Load the user's simulation module from file
module_path = "Simulation Network.py"
spec = importlib.util.spec_from_loader("sim_net", SourceFileLoader("sim_net", module_path))
sim_net = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim_net)

# Helper to compute averaged curve with confidence intervals (for both advocates and opinions)
def average_trend_with_ci(network_type, avg_degree, metric='advocates', *, agent_count=999, ticks =250, n_runs=100):
    curves = []
    for _ in range(n_runs):
        params = {}
        if network_type == "random":
            p = max(0.0, min(1.0, avg_degree / (agent_count)))  # approx avg_deg ~ p*(N-1)
            params = {"network_type": "random", "p": p}
        elif network_type == "scale_free":
            m = max(1, int(round(avg_degree/2)))
            params = {"network_type": "scale_free", "m": m}
        elif network_type == "small_world":
            k = int(round(avg_degree))
            if k % 2 == 1:  # WS requires even k
                k += 1
            params = {"network_type": "small_world", "k": k, "p": 0.1}
        else:
            raise ValueError("Unknown network type")

        model = sim_net.Model(agent_count=agent_count, **params)
        model.run(ticks=ticks)
        
        # Choose which metric to track
        if metric == 'advocates':
            curves.append(model.aware_positive_count)
        elif metric == 'opinions':
            curves.append(model.sum_opinion)
        else:
            raise ValueError("metric must be 'advocates' or 'opinions'")

    curves = np.array(curves)
    
    # Calculate mean and 95% confidence interval for the mean
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0, ddof=1)  # Sample standard deviation
    n = curves.shape[0]  # Number of simulations
    
    # Standard error of the mean
    sem_curve = std_curve / np.sqrt(n)
    
    # 95% confidence interval for the mean (t-distribution, but approximated with normal for large n)
    # For n >= 30, normal approximation is reasonable; critical value ≈ 1.96
    if n >= 30:
        critical_value = 1.96
    else:
        # For smaller samples, should use t-distribution, but using 2.0 as approximation
        critical_value = 2.0
    
    ci_lower = mean_curve - critical_value * sem_curve
    ci_upper = mean_curve + critical_value * sem_curve
    
    return mean_curve, ci_lower, ci_upper

# Set matplotlib style for professional appearance with larger fonts
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# Generate mini charts and save them
networks = [("Random","random"), ("Scale-free","scale_free"), ("Small-world","small_world")]
avg_degrees = [50, 20, 10]
deg_labels = ["links:50", "links:20", "links:10"]

image_paths = []
ticks = 250
t = np.arange(ticks+1)

# Create panel labels
panel_labels = []
for i, deg in enumerate(avg_degrees):
    for j, (net_name, ntype) in enumerate(networks):
        panel_num = i * 3 + j
        panel_labels.append(chr(ord('a') + panel_num))

# Function to create visualization for a specific metric
def create_visualization(metric='advocates'):
    image_paths = []
    
    # Set colors based on metric
    if metric == 'advocates':
        fill_color = '#4CAF50'  # Green for advocates
        line_color = '#2E7D32'
        ylabel = "Number of Advocates"
        title_suffix = "Number of Advocates Over Time"
        output_name = "network_advocates_comparison_grid_2.png"
    else:  # opinions
        fill_color = '#2196F3'  # Blue for opinions
        line_color = '#1565C0'  
        ylabel = "Sum of Expressed Opinions"
        title_suffix = "Sum of Expressed Opinions Over Time"
        output_name = "network_opinions_comparison_grid_2.png"
    
    panel_idx = 0
    for i, deg in enumerate(avg_degrees):
        for j, (net_name, ntype) in enumerate(networks):
            # Get mean curve and confidence intervals
            mean_curve, ci_lower, ci_upper = average_trend_with_ci(ntype, deg, metric=metric, agent_count=999, ticks=ticks)

            # Create a larger figure for better visibility
            fig, ax = plt.subplots(figsize=(4, 3.5))
            
            # Plot with 95% confidence interval for the mean
            ax.fill_between(t, ci_lower, ci_upper, alpha=0.3, color=fill_color, edgecolor='none', 
                           label='95% CI for mean')
            ax.plot(t, mean_curve, linewidth=2.0, color=line_color, alpha=0.95, label='Mean')
            
            # Styling to match reference
            ax.set_xlim(0, ticks)
            
            # Set Y-axis range based on metric
            if metric == 'opinions':
                ax.set_ylim(0, 65)  # Fixed range for sum of opinions
                y_ticks = [0, 30, 60]
            else:  # advocates
                y_max = max(55, mean_curve.max() * 1.2)
                ax.set_ylim(0, y_max)
                y_ticks = [0, 25, 50]
            
            # Clean tick styling with larger fonts
            ax.set_xticks([0, ticks//2, ticks])
            ax.set_xticklabels(['0', f'{ticks//2}', f'{ticks}'], fontsize=14)

            # Y-axis ticks (already set above based on metric)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{y}' for y in y_ticks], fontsize=14)
            
            # Panel label in bold - larger font
            panel_label = f"({panel_labels[panel_idx]})"
            ax.text(0.01, 1.02, panel_label, transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', va='bottom', ha='left')
            
            # Network and degree info - larger font
            ax.text(0.13, 1.02, f"{net_name}, {deg_labels[i]}", 
                   transform=ax.transAxes, fontweight='bold', fontsize=16, ha='left', va='bottom')
            
            # Axis labels only on edges - larger fonts
            if i == 2:  # Bottom row
                ax.set_xlabel("Time Step (t)", fontsize=16)
            if j == 0:  # Left column
                ax.set_ylabel(ylabel, fontsize=16)
            
            # Clean background
            ax.set_facecolor('white')
            ax.grid(True, alpha=0.2, linewidth=0.5, color='gray')
            
            # Minimal spines
            for spine in ax.spines.values():
                spine.set_color('#999999')
                spine.set_linewidth(0.6)
            
            fig.tight_layout(pad=0.3)

            fname = f"panel_{metric}_{net_name.lower().replace('-', '_')}_{deg_labels[i].replace('=', '')}.png"
            fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor='white')
            plt.close(fig)
            image_paths.append(fname)
            panel_idx += 1
    
    # Compose a professional 3x3 grid layout
    thumbs = [Image.open(p) for p in image_paths]
    w, h = thumbs[0].size

    # Create canvas with minimal spacing like reference figure
    spacing = 4
    canvas_w = w * 3 + spacing * 2
    canvas_h = h * 3 + spacing * 4
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    # Paste images with minimal spacing
    idx = 0
    for r in range(3):
        for c in range(3):
            x_pos = c * (w + spacing)
            y_pos = r * (h + spacing)
            canvas.paste(thumbs[idx], (x_pos, y_pos))
            idx += 1

    # Create final image with minimal margins (no titles/headers)
    top_margin = 20
    left_margin = 20
    right_margin = 20
    bottom_margin = 20

    final_w = canvas_w + left_margin + right_margin
    final_h = canvas_h + top_margin + bottom_margin
    final_img = Image.new("RGB", (final_w, final_h), (255, 255, 255))
    final_img.paste(canvas, (left_margin, top_margin))

    draw = ImageDraw.Draw(final_img)

    # Load fonts for title
    try:
        title_font = ImageFont.truetype("arial.ttf", 32)
        header_font = ImageFont.truetype("arial.ttf", 24)
        label_font = ImageFont.truetype("arial.ttf", 20)
    except:
        title_font = header_font = label_font = ImageFont.load_default()
    
    # No titles or headers - clean grid layout only

    # Save the final image
    final_img.save(output_name, dpi=(300, 300))
    print(f"Visualization saved as: {output_name}")
    return output_name

# Generate both visualizations
print("Generating visualizations for both metrics...")

# Generate Number of Advocates visualization (green)
advocates_output = create_visualization(metric='advocates')
print(f"✓ Advocates visualization: {advocates_output}")

# Generate Sum of Expressed Opinions visualization (blue)  
opinions_output = create_visualization(metric='opinions')
print(f"✓ Opinions visualization: {opinions_output}")

print("Both visualizations completed!")
