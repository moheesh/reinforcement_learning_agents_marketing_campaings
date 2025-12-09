"""
Main Entry Point
================
CLI interface for the Marketing Mix RL system.

Usage:
    python main.py --train           # Train everything (data + MMM + RL)
    python main.py --api             # Start API server
    python main.py --recommend       # Get recommendation
    python main.py --interactive     # Interactive mode
"""

import argparse
import json
import sys
from pathlib import Path
import yaml

from data_processor import DataProcessor
from mmm_model import MMMModel
from rl_engine import RLEngine, QLearningAgent, UCBAgent, TrainingHistory


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_allocation_table(allocation: dict, budget: float):
    """Print allocation as a formatted table."""
    print(f"\n{'Channel':<20} {'Spend':>12} {'Share':>10}")
    print("-" * 44)
    
    total = sum(allocation.values())
    for channel, spend in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
        share = spend / total * 100 if total > 0 else 0
        print(f"{channel:<20} {spend:>12,.0f} {share:>9.1f}%")
    
    print("-" * 44)
    print(f"{'Total':<20} {total:>12,.0f} {100:>9.1f}%")


def train_pipeline(args):
    """Train the full pipeline: Data → MMM → RL."""
    print_header("MARKETING MIX RL - TRAINING PIPELINE")
    
    # Step 1: Data Processing
    print_header("STEP 1: DATA PROCESSING")
    processor = DataProcessor("config.yaml")
    processor.load_data()
    processor.load_special_sales()
    processor.engineer_features()
    processor.build_state_space()
    processor.build_action_space()
    
    print(f"\n✓ Processed {processor.get_summary()['n_rows']} rows")
    print(f"✓ State space: {processor.n_states} states")
    print(f"✓ Action space: {processor.n_actions} actions")
    
    # Step 2: MMM Training
    print_header("STEP 2: MMM TRAINING")
    mmm = MMMModel("config.yaml")
    mmm.train(processor.processed_data)
    
    print(f"\n✓ R²: {mmm.metrics['r2']:.4f}")
    print(f"✓ Top channels by contribution:")
    sorted_contrib = sorted(mmm.channel_contributions.items(), key=lambda x: x[1], reverse=True)
    for ch, contrib in sorted_contrib[:5]:
        print(f"    {ch}: {contrib:.1f}%")
    
    # Add ROI-proportional action
    processor.add_roi_proportional_action(mmm.channel_rois)
    processor.save()
    mmm.save()
    
    # Step 3: RL Training
    print_header("STEP 3: RL TRAINING")
    rl = RLEngine("config.yaml")
    rl.set_environment(processor, mmm)
    
    episodes = args.episodes if hasattr(args, 'episodes') and args.episodes else 30000
    
    print(f"\nTraining Q-Learning ({episodes} episodes)...")
    q_summary = rl.train_q_learning(n_episodes=episodes, verbose=True)
    
    print(f"\nTraining UCB ({episodes} episodes)...")
    ucb_summary = rl.train_ucb(n_episodes=episodes, verbose=True)
    
    print(f"\n✓ Q-Learning mean reward: {q_summary['mean_reward']:.2f}")
    print(f"✓ UCB mean reward: {ucb_summary['mean_reward']:.2f}")
    
    # Summary
    print_header("TRAINING COMPLETE")
    print("\nSaved artifacts:")
    print("  - models/data_processor.joblib")
    print("  - models/mmm_model.joblib")
    print("  - models/q_agent.joblib")
    print("  - models/ucb_agent.joblib")
    
    print("\nNext steps:")
    print("  python main.py --recommend --budget 100 --month 7")
    print("  python main.py --api")
    print("  python main.py --interactive")


def get_recommendation(args):
    """Get a budget recommendation."""
    print_header("MARKETING MIX RL - RECOMMENDATION")
    
    # Load components
    processor = DataProcessor("config.yaml")
    processor.load("data_processor.joblib")
    processor.load_data()
    processor.engineer_features()
    
    mmm = MMMModel("config.yaml")
    mmm.load("mmm_model.joblib")
    
    rl = RLEngine("config.yaml")
    rl.set_environment(processor, mmm)
    rl.load_agents()
    
    # Get recommendation
    budget = args.budget
    month = args.month
    algo = args.algo
    nps = args.nps if hasattr(args, 'nps') and args.nps else 50
    promo = args.promo if hasattr(args, 'promo') and args.promo else 0
    
    print(f"\nInput:")
    print(f"  Budget: {budget:,.0f}")
    print(f"  Month: {month}")
    print(f"  Algorithm: {algo}")
    print(f"  NPS: {nps}")
    print(f"  Has Promotion: {promo}")
    
    # Parse constraints if provided
    constraints = None
    if hasattr(args, 'constraints') and args.constraints:
        try:
            constraints = json.loads(args.constraints)
            print(f"  Constraints: {constraints}")
        except:
            print(f"  Warning: Could not parse constraints, ignoring")
    
    rec = rl.recommend(
        budget=budget,
        month=month,
        algorithm=algo,
        nps=nps,
        has_promotion=promo,
        constraints=constraints
    )
    
    if algo == 'compare':
        # Compare mode
        print_header("Q-LEARNING RECOMMENDATION")
        q_rec = rec['q_learning']
        print(f"\nAction: {q_rec['action_name']}")
        print(f"Description: {q_rec['action_description']}")
        print(f"Predicted Revenue: {q_rec['predicted_revenue']:,.0f}")
        print(f"Predicted ROAS: {q_rec['predicted_roas']:.2f}")
        print_allocation_table(q_rec['allocation'], budget)
        
        print_header("UCB RECOMMENDATION")
        ucb_rec = rec['ucb']
        print(f"\nAction: {ucb_rec['action_name']}")
        print(f"Description: {ucb_rec['action_description']}")
        print(f"Predicted Revenue: {ucb_rec['predicted_revenue']:,.0f}")
        print(f"Predicted ROAS: {ucb_rec['predicted_roas']:.2f}")
        print_allocation_table(ucb_rec['allocation'], budget)
        
        print_header("COMPARISON")
        diff = rec['comparison']['revenue_diff']
        print(f"\nRevenue difference (Q - UCB): {diff:,.0f}")
        print(f"Same action: {rec['comparison']['same_action']}")
        
        if diff > 0:
            print("\n→ Q-Learning recommends higher revenue allocation")
        elif diff < 0:
            print("\n→ UCB recommends higher revenue allocation")
        else:
            print("\n→ Both algorithms recommend same allocation")
    
    else:
        # Single algorithm mode
        print_header(f"{algo.upper()} RECOMMENDATION")
        print(f"\nAction: {rec['action_name']}")
        print(f"Description: {rec['action_description']}")
        print(f"Predicted Revenue: {rec['predicted_revenue']:,.0f}")
        print(f"Predicted ROAS: {rec['predicted_roas']:.2f}")
        print_allocation_table(rec['allocation'], budget)


def run_interactive(args):
    """Run interactive mode."""
    print_header("MARKETING MIX RL - INTERACTIVE MODE")
    
    # Load components
    print("\nLoading components...")
    processor = DataProcessor("config.yaml")
    processor.load("data_processor.joblib")
    processor.load_data()
    processor.engineer_features()
    
    mmm = MMMModel("config.yaml")
    mmm.load("mmm_model.joblib")
    
    rl = RLEngine("config.yaml")
    rl.set_environment(processor, mmm)
    rl.load_agents()
    
    print("✓ All components loaded")
    print("\nCommands:")
    print("  recommend <budget> <month>  - Get recommendation")
    print("  compare <budget> <month>    - Compare Q vs UCB")
    print("  whatif <from> <to> <amount> - What-if analysis")
    print("  marginal                    - Show marginal ROI")
    print("  actions                     - List all actions")
    print("  policy q|ucb                - Show learned policy")
    print("  help                        - Show commands")
    print("  quit                        - Exit")
    
    # Default context
    context = {'nps': 50, 'month': 6, 'has_promotion': 0}
    current_allocation = {ch: 10 for ch in processor.channels if ch in processor.processed_data.columns}
    
    while True:
        try:
            user_input = input("\n> ").strip().lower()
            
            if not user_input:
                continue
            
            parts = user_input.split()
            cmd = parts[0]
            
            if cmd == 'quit' or cmd == 'exit' or cmd == 'q':
                print("Goodbye!")
                break
            
            elif cmd == 'help':
                print("\nCommands:")
                print("  recommend <budget> <month>  - Get recommendation")
                print("  compare <budget> <month>    - Compare Q vs UCB")
                print("  whatif <from> <to> <amount> - What-if analysis")
                print("  marginal                    - Show marginal ROI")
                print("  actions                     - List all actions")
                print("  policy q|ucb                - Show learned policy")
                print("  quit                        - Exit")
            
            elif cmd == 'recommend':
                if len(parts) < 3:
                    print("Usage: recommend <budget> <month>")
                    continue
                
                budget = float(parts[1])
                month = int(parts[2])
                algo = parts[3] if len(parts) > 3 else 'ucb'
                
                rec = rl.recommend(budget=budget, month=month, algorithm=algo)
                
                print(f"\nAction: {rec['action_name']}")
                print(f"Predicted Revenue: {rec['predicted_revenue']:,.0f}")
                print_allocation_table(rec['allocation'], budget)
                
                current_allocation = rec['allocation']
            
            elif cmd == 'compare':
                if len(parts) < 3:
                    print("Usage: compare <budget> <month>")
                    continue
                
                budget = float(parts[1])
                month = int(parts[2])
                
                rec = rl.recommend(budget=budget, month=month, algorithm='compare')
                
                print(f"\nQ-Learning: {rec['q_learning']['action_name']}")
                print(f"  Revenue: {rec['q_learning']['predicted_revenue']:,.0f}")
                
                print(f"\nUCB: {rec['ucb']['action_name']}")
                print(f"  Revenue: {rec['ucb']['predicted_revenue']:,.0f}")
                
                diff = rec['comparison']['revenue_diff']
                winner = "Q-Learning" if diff > 0 else "UCB" if diff < 0 else "Tie"
                print(f"\nBetter: {winner} (diff: {abs(diff):,.0f})")
            
            elif cmd == 'whatif':
                if len(parts) < 4:
                    print("Usage: whatif <from_channel> <to_channel> <amount>")
                    print(f"Channels: {list(current_allocation.keys())}")
                    continue
                
                from_ch = parts[1]
                to_ch = parts[2]
                amount = float(parts[3])
                
                # Find matching channel names (case insensitive)
                channels = list(current_allocation.keys())
                from_match = next((c for c in channels if c.lower().startswith(from_ch)), None)
                to_match = next((c for c in channels if c.lower().startswith(to_ch)), None)
                
                if not from_match or not to_match:
                    print(f"Channel not found. Available: {channels}")
                    continue
                
                result = mmm.simulate_reallocation(
                    current_allocation, from_match, to_match, amount, context
                )
                
                print(f"\nMoving {amount:,.0f} from {from_match} to {to_match}")
                print(f"Current Revenue: {result['current_revenue']:,.0f}")
                print(f"New Revenue: {result['new_revenue']:,.0f}")
                print(f"Change: {result['revenue_change']:+,.0f} ({result['revenue_change_pct']:+.2f}%)")
            
            elif cmd == 'marginal':
                marginal = mmm.get_marginal_roi(current_allocation, context)
                
                print("\nMarginal ROI (revenue per $1 additional spend):")
                print(f"{'Channel':<20} {'Marginal ROI':>12}")
                print("-" * 34)
                
                for ch, roi in sorted(marginal.items(), key=lambda x: x[1], reverse=True):
                    print(f"{ch:<20} {roi:>12.4f}")
            
            elif cmd == 'actions':
                print("\nAvailable Actions:")
                for action in processor.actions:
                    print(f"  [{action['id']}] {action['name']}")
                    print(f"      {action['description']}")
            
            elif cmd == 'policy':
                if len(parts) < 2:
                    print("Usage: policy q|ucb")
                    continue
                
                algo = parts[1]
                summary = rl.get_policy_summary(algo)
                
                if 'error' in summary:
                    print(f"Error: {summary['error']}")
                    continue
                
                if algo == 'ucb':
                    print("\nUCB Policy by Month:")
                    for month, action in summary['policy_by_month'].items():
                        print(f"  {month}: {action}")
                else:
                    print("\nQ-Learning Action Distribution:")
                    for action, count in summary['action_distribution'].items():
                        if count > 0:
                            print(f"  {action}: {count} states")
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def start_api(args):
    """Start the API server."""
    print_header("MARKETING MIX RL - API SERVER")
    
    import uvicorn
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    api_config = config.get('api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    
    print(f"\nStarting server...")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Docs: http://localhost:{port}/docs")
    print(f"  Health: http://localhost:{port}/health")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=api_config.get('reload', True)
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Marketing Mix Optimization with Reinforcement Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train                          # Train full pipeline
  python main.py --train --episodes 50000         # Train with more episodes
  python main.py --api                            # Start API server
  python main.py --recommend --budget 100 --month 7
  python main.py --recommend --budget 200 --month 12 --algo compare
  python main.py --interactive                    # Interactive mode
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train the full pipeline')
    mode_group.add_argument('--api', action='store_true', help='Start API server')
    mode_group.add_argument('--recommend', action='store_true', help='Get recommendation')
    mode_group.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    # Training options
    parser.add_argument('--episodes', type=int, default=30000, help='RL training episodes')
    
    # Recommendation options
    parser.add_argument('--budget', type=float, default=100, help='Budget to allocate')
    parser.add_argument('--month', type=int, default=7, help='Month (1-12)')
    parser.add_argument('--algo', type=str, default='compare', 
                       choices=['q', 'ucb', 'compare'], help='Algorithm to use')
    parser.add_argument('--nps', type=float, default=50, help='NPS score')
    parser.add_argument('--promo', type=int, default=0, help='Has promotion (0 or 1)')
    parser.add_argument('--constraints', type=str, default=None,
                       help='JSON constraints, e.g. \'{"TV": {"min": 10}}\'')
    
    args = parser.parse_args()
    
    # Check if models exist for recommend/interactive modes
    models_path = Path("models")
    required_files = [
        "data_processor.joblib",
        "mmm_model.joblib", 
        "q_agent.joblib",
        "ucb_agent.joblib"
    ]
    
    if args.recommend or args.interactive:
        missing = [f for f in required_files if not (models_path / f).exists()]
        if missing:
            print("Error: Models not found. Run training first:")
            print("  python main.py --train")
            print(f"\nMissing files: {missing}")
            sys.exit(1)
    
    # Execute selected mode
    if args.train:
        train_pipeline(args)
    elif args.api:
        start_api(args)
    elif args.recommend:
        get_recommendation(args)
    elif args.interactive:
        run_interactive(args)


if __name__ == "__main__":
    main()