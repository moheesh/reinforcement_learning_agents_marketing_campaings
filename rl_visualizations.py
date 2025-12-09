"""
Marketing Mix RL Visualizations
===============================
Generate and save visualizations for Data Processing, MMM, and RL training.
Run after training is complete.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import joblib
import yaml

def create_visualizations(config_path: str = "config.yaml"):
    """Generate all visualizations and save to outputs folder."""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    models_path = Path(config['paths']['models'])
    outputs_path = Path(config['paths'].get('outputs', 'outputs'))
    outputs_path.mkdir(exist_ok=True)
    
    # Load all saved models/data
    q_state = _load_safe(models_path / "q_agent.joblib")
    ucb_state = _load_safe(models_path / "ucb_agent.joblib")
    mmm_state = _load_safe(models_path / "mmm_model.joblib")
    processor_state = _load_safe(models_path / "data_processor.joblib")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Generate all visualizations
    if q_state or ucb_state:
        _viz_training_rewards(q_state, ucb_state, outputs_path)
        _viz_qlearning_analysis(q_state, outputs_path)
        _viz_ucb_analysis(ucb_state, outputs_path)
        _viz_algorithm_comparison(q_state, ucb_state, outputs_path)
    
    if mmm_state:
        _viz_mmm_channel_analysis(mmm_state, outputs_path)
        _viz_mmm_model_performance(mmm_state, outputs_path)
    
    if processor_state:
        _viz_action_space(processor_state, outputs_path)
        _viz_state_space(processor_state, outputs_path)
    
    print(f"\n✓ All visualizations saved to {outputs_path}/")


def _load_safe(path):
    """Safely load joblib file."""
    try:
        return joblib.load(path) if path.exists() else None
    except:
        return None


# =============================================================================
# RL VISUALIZATIONS
# =============================================================================

def _viz_training_rewards(q_state, ucb_state, outputs_path):
    """Training reward curves for both algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if q_state:
        rewards = q_state['history'].rewards
        window = min(500, len(rewards)//10 or 1)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        axes[0].plot(smoothed, color='#2E86AB', linewidth=1.5, alpha=0.8)
        axes[0].fill_between(range(len(smoothed)), smoothed, alpha=0.3, color='#2E86AB')
        axes[0].axhline(y=np.mean(rewards), color='#E63946', linestyle='--', 
                        label=f'Mean: {np.mean(rewards):.4f}', alpha=0.7)
        axes[0].set_title('Q-Learning Training Rewards', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Episode', fontsize=11)
        axes[0].set_ylabel('Reward (Moving Avg)', fontsize=11)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'Q-Learning not trained', ha='center', va='center', fontsize=12)
        axes[0].set_title('Q-Learning Training Rewards', fontsize=14, fontweight='bold')
    
    if ucb_state:
        rewards = ucb_state['history'].rewards
        window = min(500, len(rewards)//10 or 1)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        axes[1].plot(smoothed, color='#A23B72', linewidth=1.5, alpha=0.8)
        axes[1].fill_between(range(len(smoothed)), smoothed, alpha=0.3, color='#A23B72')
        axes[1].axhline(y=np.mean(rewards), color='#E63946', linestyle='--',
                        label=f'Mean: {np.mean(rewards):.4f}', alpha=0.7)
        axes[1].set_title('UCB Training Rewards', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Episode', fontsize=11)
        axes[1].set_ylabel('Reward (Moving Avg)', fontsize=11)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'UCB not trained', ha='center', va='center', fontsize=12)
        axes[1].set_title('UCB Training Rewards', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(outputs_path / 'rl_training_rewards.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: rl_training_rewards.png")


def _viz_qlearning_analysis(q_state, outputs_path):
    """Q-Learning epsilon decay and action distribution."""
    if not q_state:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Epsilon decay
    epsilons = q_state['history'].epsilon_values
    axes[0].plot(epsilons, color='#F18F01', linewidth=1.5)
    axes[0].set_title('Epsilon Decay (Exploration → Exploitation)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Episode', fontsize=11)
    axes[0].set_ylabel('Epsilon', fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=epsilons[-1], color='#E63946', linestyle='--', 
                    label=f'Final: {epsilons[-1]:.4f}')
    axes[0].legend()
    
    # Action distribution
    actions = q_state['history'].actions
    unique, counts = np.unique(actions, return_counts=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(unique)))
    bars = axes[1].bar(unique, counts, color=colors, edgecolor='white', linewidth=0.5)
    axes[1].set_title('Action Selection Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Action ID', fontsize=11)
    axes[1].set_ylabel('Selection Count', fontsize=11)
    
    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        if pct > 5:
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(outputs_path / 'rl_qlearning_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: rl_qlearning_analysis.png")


def _viz_ucb_analysis(ucb_state, outputs_path):
    """UCB visit counts and Q-value heatmaps."""
    if not ucb_state:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Visit counts heatmap
    N = ucb_state['N']
    im = axes[0].imshow(N, cmap='YlOrRd', aspect='auto')
    axes[0].set_title('UCB Visit Counts (Context × Action)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Action ID', fontsize=11)
    axes[0].set_ylabel('Context (Month)', fontsize=11)
    axes[0].set_yticks(range(min(12, N.shape[0])))
    axes[0].set_yticklabels(month_labels[:N.shape[0]])
    plt.colorbar(im, ax=axes[0], label='Visit Count')
    
    # Q-value heatmap
    Q = ucb_state['Q']
    im2 = axes[1].imshow(Q, cmap='RdYlGn', aspect='auto')
    axes[1].set_title('Learned Q-Values (Context × Action)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Action ID', fontsize=11)
    axes[1].set_ylabel('Context (Month)', fontsize=11)
    axes[1].set_yticks(range(min(12, Q.shape[0])))
    axes[1].set_yticklabels(month_labels[:Q.shape[0]])
    plt.colorbar(im2, ax=axes[1], label='Q-Value')
    
    # Mark best action per context
    best_actions = np.argmax(Q, axis=1)
    for ctx, action in enumerate(best_actions):
        axes[1].scatter(action, ctx, marker='*', s=200, c='black', zorder=5)
    
    plt.tight_layout()
    plt.savefig(outputs_path / 'rl_ucb_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: rl_ucb_analysis.png")


def _viz_algorithm_comparison(q_state, ucb_state, outputs_path):
    """Compare Q-Learning and UCB performance."""
    if not q_state or not ucb_state:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    q_rewards = q_state['history'].rewards
    ucb_rewards = ucb_state['history'].rewards
    
    # Reward distributions
    axes[0, 0].hist(q_rewards, bins=50, alpha=0.6, color='#2E86AB', label='Q-Learning', density=True)
    axes[0, 0].hist(ucb_rewards, bins=50, alpha=0.6, color='#A23B72', label='UCB', density=True)
    axes[0, 0].set_title('Reward Distribution Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    
    # Cumulative reward
    axes[0, 1].plot(np.cumsum(q_rewards), label='Q-Learning', color='#2E86AB', linewidth=1.5)
    axes[0, 1].plot(np.cumsum(ucb_rewards), label='UCB', color='#A23B72', linewidth=1.5)
    axes[0, 1].set_title('Cumulative Reward Over Training', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Cumulative Reward')
    axes[0, 1].legend()
    
    # Rolling mean comparison
    window = 1000
    q_rolling = pd.Series(q_rewards).rolling(window).mean()
    ucb_rolling = pd.Series(ucb_rewards).rolling(window).mean()
    axes[1, 0].plot(q_rolling, label='Q-Learning', color='#2E86AB', linewidth=1.5)
    axes[1, 0].plot(ucb_rolling, label='UCB', color='#A23B72', linewidth=1.5)
    axes[1, 0].set_title(f'Rolling Mean Reward (window={window})', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Mean Reward')
    axes[1, 0].legend()
    
    # Summary statistics
    metrics = ['Mean', 'Std', 'Max', 'Min']
    q_stats = [np.mean(q_rewards), np.std(q_rewards), np.max(q_rewards), np.min(q_rewards)]
    ucb_stats = [np.mean(ucb_rewards), np.std(ucb_rewards), np.max(ucb_rewards), np.min(ucb_rewards)]
    
    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 1].bar(x - width/2, q_stats, width, label='Q-Learning', color='#2E86AB')
    axes[1, 1].bar(x + width/2, ucb_stats, width, label='UCB', color='#A23B72')
    axes[1, 1].set_title('Training Statistics', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(outputs_path / 'rl_algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: rl_algorithm_comparison.png")


# =============================================================================
# MMM VISUALIZATIONS
# =============================================================================

def _viz_mmm_channel_analysis(mmm_state, outputs_path):
    """MMM channel contributions, ROI, and adstock analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Channel Contributions (Pie)
    contributions = mmm_state.get('channel_contributions', {})
    if contributions:
        channels = list(contributions.keys())
        values = list(contributions.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(channels)))
        
        wedges, texts, autotexts = axes[0, 0].pie(
            values, labels=channels, autopct='%1.1f%%', colors=colors,
            explode=[0.02]*len(channels), shadow=True
        )
        axes[0, 0].set_title('Channel Contribution to Revenue', fontsize=12, fontweight='bold')
    
    # Channel ROI (Bar)
    rois = mmm_state.get('channel_rois', {})
    if rois:
        channels = list(rois.keys())
        values = list(rois.values())
        colors = ['#2E86AB' if v >= 0 else '#E63946' for v in values]
        
        bars = axes[0, 1].barh(channels, values, color=colors, edgecolor='white')
        axes[0, 1].axvline(x=0, color='black', linewidth=0.5)
        axes[0, 1].set_title('Channel ROI (Revenue per $ Spent)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('ROI')
        
        for bar, val in zip(bars, values):
            axes[0, 1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', fontsize=9)
    
    # Adstock Decay Rates
    adstock = mmm_state.get('adstock_decays', {})
    if adstock:
        channels = list(adstock.keys())
        values = list(adstock.values())
        colors = plt.cm.coolwarm(np.array(values))
        
        bars = axes[1, 0].bar(channels, values, color=colors, edgecolor='white')
        axes[1, 0].set_title('Adstock Decay Rates by Channel', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Decay Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars, values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Model Coefficients
    coefficients = mmm_state.get('coefficients', {})
    if coefficients:
        # Filter to channel coefficients only
        channel_coefs = {k: v for k, v in coefficients.items() 
                        if k not in ['nps', 'NPS', 'has_promotion', 'month_sin', 'month_cos', 'trend']}
        
        if channel_coefs:
            channels = list(channel_coefs.keys())
            values = list(channel_coefs.values())
            colors = ['#2E86AB' if v >= 0 else '#E63946' for v in values]
            
            axes[1, 1].barh(channels, values, color=colors, edgecolor='white')
            axes[1, 1].axvline(x=0, color='black', linewidth=0.5)
            axes[1, 1].set_title('Model Coefficients (Scaled)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Coefficient')
    
    plt.tight_layout()
    plt.savefig(outputs_path / 'mmm_channel_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: mmm_channel_analysis.png")


def _viz_mmm_model_performance(mmm_state, outputs_path):
    """MMM model performance metrics and saturation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Model Metrics
    metrics = mmm_state.get('metrics', {})
    if metrics:
        metric_names = ['R²', 'MAPE (%)', 'RMSE', 'CV R² Mean']
        metric_values = [
            metrics.get('r2', 0),
            metrics.get('mape', 0),
            metrics.get('rmse', 0) / 1000,  # Scale down for display
            metrics.get('cv_r2_mean', 0)
        ]
        
        colors = ['#2E86AB', '#F18F01', '#A23B72', '#2E86AB']
        bars = axes[0].bar(metric_names, metric_values, color=colors, edgecolor='white')
        axes[0].set_title('MMM Model Performance Metrics', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Value')
        
        for bar, val, name in zip(bars, metric_values, metric_names):
            display_val = val * 1000 if 'RMSE' in name else val
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{display_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Saturation Curves Illustration
    saturation_params = mmm_state.get('saturation_params', {})
    if saturation_params:
        x = np.linspace(0, 100, 200)
        
        for i, (channel, half_sat) in enumerate(list(saturation_params.items())[:5]):
            y = x / (x + half_sat + 1e-10)
            axes[1].plot(x, y, label=f'{channel} (λ={half_sat:.1f})', linewidth=2)
        
        axes[1].set_title('Saturation Curves (Diminishing Returns)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Spend')
        axes[1].set_ylabel('Saturated Effect')
        axes[1].legend(loc='lower right')
        axes[1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(outputs_path / 'mmm_model_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: mmm_model_performance.png")


# =============================================================================
# DATA PROCESSOR VISUALIZATIONS
# =============================================================================

def _viz_action_space(processor_state, outputs_path):
    """Visualize action space allocations."""
    if not processor_state:
        return
    
    actions = processor_state.get('actions', [])
    if not actions:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Action allocation heatmap
    allocation_templates = processor_state.get('allocation_templates')
    channels = processor_state.get('channels', [])
    
    if allocation_templates is not None and len(allocation_templates) > 0:
        im = axes[0].imshow(allocation_templates.T, cmap='YlGnBu', aspect='auto')
        axes[0].set_title('Action Space: Budget Allocation Templates', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Action ID')
        axes[0].set_ylabel('Channel')
        
        if len(channels) <= allocation_templates.shape[1]:
            axes[0].set_yticks(range(len(channels)))
            axes[0].set_yticklabels(channels, fontsize=8)
        
        axes[0].set_xticks(range(len(actions)))
        plt.colorbar(im, ax=axes[0], label='Allocation %')
    
    # Action types distribution
    action_sources = [a.get('source', 'unknown') for a in actions]
    unique_sources, counts = np.unique(action_sources, return_counts=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_sources)))
    
    axes[1].pie(counts, labels=unique_sources, autopct='%1.0f%%', colors=colors,
                explode=[0.02]*len(unique_sources))
    axes[1].set_title('Action Sources Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(outputs_path / 'data_action_space.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: data_action_space.png")


def _viz_state_space(processor_state, outputs_path):
    """Visualize state space configuration."""
    if not processor_state:
        return
    
    state_config = processor_state.get('state_config', {})
    state_bins = processor_state.get('state_bins', {})
    
    if not state_config or not state_bins:
        return
    
    dimensions = state_config.get('dimensions', [])
    if not dimensions:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # State dimensions and bins
    dim_names = []
    bin_counts = []
    
    for dim in dimensions:
        if dim in state_bins:
            dim_names.append(dim.replace('_', '\n'))
            bin_counts.append(state_bins[dim]['bins'])
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(dim_names)))
    bars = axes[0].bar(dim_names, bin_counts, color=colors, edgecolor='white')
    axes[0].set_title('State Space Dimensions', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Bins')
    
    for bar, count in zip(bars, bin_counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # State space summary
    n_states = processor_state.get('n_states', 0)
    n_actions = processor_state.get('n_actions', 0)
    
    summary_text = f"""State Space Summary
    
Total States: {n_states:,}
Total Actions: {n_actions}
State-Action Pairs: {n_states * n_actions:,}

Dimensions: {len(dimensions)}
{chr(10).join(f'  • {d}: {state_bins[d]["bins"]} bins' for d in dimensions if d in state_bins)}
"""
    
    axes[1].text(0.1, 0.5, summary_text, transform=axes[1].transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    axes[1].axis('off')
    axes[1].set_title('Configuration Summary', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(outputs_path / 'data_state_space.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: data_state_space.png")


if __name__ == "__main__":
    create_visualizations()