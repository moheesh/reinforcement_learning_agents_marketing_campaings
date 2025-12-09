"""
Marketing Mix RL - Budget Allocation Analysis
==============================================
Tests budget allocations across different conditions and displays
detailed channel split-ups.

Run: python test_allocations.py
Make sure API is running: python main.py --api
"""

import requests
import json
from typing import Dict, List
from tabulate import tabulate  # pip install tabulate

BASE_URL = "http://localhost:8000"

# Channel display order
CHANNELS = [
    'TV', 'Digital', 'Sponsorship', 'Content.Marketing', 
    'Online.marketing', 'Affiliates', 'SEM', 'Radio', 'Other'
]


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subheader(title: str):
    print(f"\n{'─' * 40}")
    print(f" {title}")
    print(f"{'─' * 40}")


def get_recommendation(budget: float, month: int, algorithm: str = "compare", 
                       nps: float = 50, has_promotion: int = 0) -> Dict:
    """Get recommendation from API."""
    payload = {
        "budget": budget,
        "month": month,
        "algorithm": algorithm,
        "nps": nps,
        "has_promotion": has_promotion
    }
    response = requests.post(f"{BASE_URL}/recommend", json=payload)
    return response.json()


def format_currency(value: float) -> str:
    """Format as currency."""
    if value >= 1000000:
        return f"${value/1000000:.1f}M"
    elif value >= 1000:
        return f"${value/1000:.1f}K"
    else:
        return f"${value:.0f}"


def format_allocation_table(allocation: Dict, budget: float) -> str:
    """Create formatted allocation table."""
    rows = []
    for ch in CHANNELS:
        spend = allocation.get(ch, 0)
        if spend > 0 or ch in allocation:
            pct = (spend / budget * 100) if budget > 0 else 0
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            rows.append([ch, f"${spend:,.0f}", f"{pct:.1f}%", bar[:25]])
    
    return tabulate(rows, headers=["Channel", "Spend", "Share", "Distribution"], 
                   tablefmt="simple", colalign=("left", "right", "right", "left"))


def print_recommendation(rec: Dict, title: str, budget: float):
    """Print a single recommendation nicely."""
    print(f"\n📊 {title}")
    print(f"   Strategy: {rec.get('action_name', 'N/A')}")
    print(f"   Description: {rec.get('action_description', 'N/A')}")
    print(f"   Predicted Revenue: {format_currency(rec.get('predicted_revenue', 0))}")
    print(f"   ROAS: {rec.get('predicted_roas', 0):,.2f}x")
    print()
    print(format_allocation_table(rec.get('allocation', {}), budget))


def test_varying_budgets():
    """Test 1: How allocations change with different budget levels."""
    print_header("TEST 1: BUDGET VARIATIONS")
    print("How does the optimal allocation change as budget increases?")
    
    budgets = [50000, 100000, 200000, 500000, 1000000]
    month = 7  # July
    
    results_q = []
    results_ucb = []
    
    for budget in budgets:
        data = get_recommendation(budget, month, "compare")
        if data.get('success'):
            rec = data['recommendation']
            
            q = rec['q_learning']
            ucb = rec['ucb']
            
            results_q.append({
                'budget': budget,
                'strategy': q['action_name'],
                'revenue': q['predicted_revenue'],
                'allocation': q['allocation']
            })
            
            results_ucb.append({
                'budget': budget,
                'strategy': ucb['action_name'],
                'revenue': ucb['predicted_revenue'],
                'allocation': ucb['allocation']
            })
    
    # Q-Learning Results
    print_subheader("Q-LEARNING ALLOCATIONS BY BUDGET")
    
    for r in results_q:
        print(f"\n💰 Budget: {format_currency(r['budget'])}")
        print(f"   Strategy: {r['strategy']}")
        print(f"   Revenue: {format_currency(r['revenue'])}")
        print()
        print(format_allocation_table(r['allocation'], r['budget']))
    
    # UCB Results
    print_subheader("UCB ALLOCATIONS BY BUDGET")
    
    for r in results_ucb:
        print(f"\n💰 Budget: {format_currency(r['budget'])}")
        print(f"   Strategy: {r['strategy']}")
        print(f"   Revenue: {format_currency(r['revenue'])}")
        print()
        print(format_allocation_table(r['allocation'], r['budget']))
    
    # Comparison Table
    print_subheader("BUDGET COMPARISON SUMMARY")
    
    headers = ["Budget", "Q-Learning Strategy", "Q Revenue", "UCB Strategy", "UCB Revenue", "Winner"]
    rows = []
    for q, ucb in zip(results_q, results_ucb):
        winner = "Q-Learning" if q['revenue'] > ucb['revenue'] else "UCB" if ucb['revenue'] > q['revenue'] else "Tie"
        rows.append([
            format_currency(q['budget']),
            q['strategy'][:20],
            format_currency(q['revenue']),
            ucb['strategy'][:20],
            format_currency(ucb['revenue']),
            winner
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def test_varying_months():
    """Test 2: How allocations change across different months/seasons."""
    print_header("TEST 2: SEASONAL VARIATIONS")
    print("How does the optimal allocation change throughout the year?")
    
    budget = 100000
    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    results = []
    
    for month in months:
        data = get_recommendation(budget, month, "compare")
        if data.get('success'):
            rec = data['recommendation']
            ucb = rec['ucb']  # Use UCB as primary
            
            results.append({
                'month': month,
                'month_name': month_names[month-1],
                'strategy': ucb['action_name'],
                'revenue': ucb['predicted_revenue'],
                'allocation': ucb['allocation']
            })
    
    # Monthly breakdown
    print_subheader("UCB ALLOCATIONS BY MONTH (Budget: $100K)")
    
    # Create allocation matrix
    headers = ["Month"] + CHANNELS + ["Revenue"]
    rows = []
    
    for r in results:
        row = [r['month_name']]
        for ch in CHANNELS:
            spend = r['allocation'].get(ch, 0)
            pct = spend / budget * 100
            row.append(f"{pct:.0f}%")
        row.append(format_currency(r['revenue']))
        rows.append(row)
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Seasonal summary
    print_subheader("SEASONAL SUMMARY")
    
    seasons = {
        'Q1 (Jan-Mar)': [0, 1, 2],
        'Q2 (Apr-Jun)': [3, 4, 5],
        'Q3 (Jul-Sep)': [6, 7, 8],
        'Q4 (Oct-Dec)': [9, 10, 11]
    }
    
    for season_name, indices in seasons.items():
        season_results = [results[i] for i in indices]
        avg_revenue = sum(r['revenue'] for r in season_results) / len(season_results)
        strategies = set(r['strategy'] for r in season_results)
        
        print(f"\n{season_name}")
        print(f"  Avg Revenue: {format_currency(avg_revenue)}")
        print(f"  Strategies Used: {', '.join(strategies)}")


def test_promotion_impact():
    """Test 3: How promotions affect allocation."""
    print_header("TEST 3: PROMOTION IMPACT")
    print("How does having a promotion affect the optimal allocation?")
    
    budget = 100000
    month = 11  # November (typical promo month)
    
    # Without promotion
    data_no_promo = get_recommendation(budget, month, "compare", has_promotion=0)
    # With promotion
    data_promo = get_recommendation(budget, month, "compare", has_promotion=1)
    
    if data_no_promo.get('success') and data_promo.get('success'):
        no_promo = data_no_promo['recommendation']['ucb']
        with_promo = data_promo['recommendation']['ucb']
        
        print_subheader("WITHOUT PROMOTION")
        print_recommendation(no_promo, "No Promotion", budget)
        
        print_subheader("WITH PROMOTION")
        print_recommendation(with_promo, "With Promotion", budget)
        
        print_subheader("ALLOCATION CHANGES")
        
        headers = ["Channel", "No Promo", "With Promo", "Change"]
        rows = []
        
        for ch in CHANNELS:
            no_p = no_promo['allocation'].get(ch, 0)
            with_p = with_promo['allocation'].get(ch, 0)
            change = with_p - no_p
            change_str = f"+${change:,.0f}" if change > 0 else f"-${abs(change):,.0f}" if change < 0 else "$0"
            
            rows.append([ch, f"${no_p:,.0f}", f"${with_p:,.0f}", change_str])
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        print(f"\nRevenue Impact:")
        print(f"  Without Promo: {format_currency(no_promo['predicted_revenue'])}")
        print(f"  With Promo: {format_currency(with_promo['predicted_revenue'])}")
        print(f"  Difference: {format_currency(with_promo['predicted_revenue'] - no_promo['predicted_revenue'])}")


def test_nps_impact():
    """Test 4: How NPS score affects allocation."""
    print_header("TEST 4: NPS SCORE IMPACT")
    print("How does customer satisfaction (NPS) affect the optimal allocation?")
    
    budget = 100000
    month = 7
    nps_levels = [30, 45, 50, 60, 70]  # Low to High NPS
    
    results = []
    
    for nps in nps_levels:
        data = get_recommendation(budget, month, "ucb", nps=nps)
        if data.get('success'):
            rec = data['recommendation']
            results.append({
                'nps': nps,
                'strategy': rec['action_name'],
                'revenue': rec['predicted_revenue'],
                'allocation': rec['allocation']
            })
    
    print_subheader("ALLOCATIONS BY NPS SCORE")
    
    headers = ["NPS"] + CHANNELS + ["Revenue"]
    rows = []
    
    for r in results:
        row = [f"{r['nps']}"]
        for ch in CHANNELS:
            spend = r['allocation'].get(ch, 0)
            row.append(f"${spend/1000:.0f}K")
        row.append(format_currency(r['revenue']))
        rows.append(row)
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def test_algorithm_comparison():
    """Test 5: Detailed Q-Learning vs UCB comparison."""
    print_header("TEST 5: Q-LEARNING vs UCB DETAILED COMPARISON")
    print("Side-by-side comparison of both algorithms")
    
    budget = 100000
    month = 7
    
    data = get_recommendation(budget, month, "compare")
    
    if data.get('success'):
        rec = data['recommendation']
        q = rec['q_learning']
        ucb = rec['ucb']
        comp = rec['comparison']
        
        print(f"\nBudget: {format_currency(budget)} | Month: July")
        
        print_subheader("Q-LEARNING RECOMMENDATION")
        print_recommendation(q, "Q-Learning (Value-Based RL)", budget)
        
        print_subheader("UCB RECOMMENDATION")
        print_recommendation(ucb, "UCB (Exploration Strategy)", budget)
        
        print_subheader("SIDE-BY-SIDE ALLOCATION COMPARISON")
        
        headers = ["Channel", "Q-Learning", "Q %", "UCB", "UCB %", "Difference"]
        rows = []
        
        for ch in CHANNELS:
            q_spend = q['allocation'].get(ch, 0)
            ucb_spend = ucb['allocation'].get(ch, 0)
            q_pct = q_spend / budget * 100
            ucb_pct = ucb_spend / budget * 100
            diff = ucb_spend - q_spend
            diff_str = f"+${diff:,.0f}" if diff > 0 else f"-${abs(diff):,.0f}" if diff < 0 else "—"
            
            rows.append([
                ch, 
                f"${q_spend:,.0f}", f"{q_pct:.1f}%",
                f"${ucb_spend:,.0f}", f"{ucb_pct:.1f}%",
                diff_str
            ])
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        print_subheader("PERFORMANCE COMPARISON")
        
        winner = "Q-Learning" if comp['revenue_diff'] > 0 else "UCB" if comp['revenue_diff'] < 0 else "Tie"
        
        print(f"""
┌────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE SUMMARY                         │
├────────────────────────────────────────────────────────────────┤
│  Q-Learning Revenue:  {format_currency(q['predicted_revenue']):>15}                       │
│  UCB Revenue:         {format_currency(ucb['predicted_revenue']):>15}                       │
│  Revenue Difference:  {format_currency(abs(comp['revenue_diff'])):>15}                       │
│  Same Strategy:       {'Yes' if comp['same_action'] else 'No':>15}                       │
│                                                                │
│  🏆 WINNER: {winner:^20}                            │
└────────────────────────────────────────────────────────────────┘
""")


def test_all_strategies():
    """Test 6: Show all available strategies and their allocations."""
    print_header("TEST 6: ALL AVAILABLE STRATEGIES")
    print("Complete breakdown of all discovered allocation strategies")
    
    response = requests.get(f"{BASE_URL}/data/actions")
    data = response.json()
    
    if data.get('success'):
        actions = data['actions']
        
        budget = 100000  # Reference budget
        
        print_subheader("CLUSTER-BASED STRATEGIES (Discovered from Historical Data)")
        
        for action in actions:
            if action.get('source') == 'cluster':
                print(f"\n[{action['id']}] {action['name']}")
                print(f"    {action['description']}")
                
                allocation = action['allocation']
                
                # Sort by share
                sorted_alloc = sorted(allocation.items(), key=lambda x: x[1], reverse=True)
                
                print(f"\n    Allocation (for ${budget/1000:.0f}K budget):")
                for ch, share in sorted_alloc:
                    if share > 0.01:  # Only show > 1%
                        spend = share * budget
                        bar = "█" * int(share * 40)
                        print(f"    {ch:20s} ${spend:>8,.0f} ({share*100:5.1f}%) {bar}")
        
        print_subheader("GENERATED STRATEGIES")
        
        for action in actions:
            if action.get('source') != 'cluster':
                print(f"\n[{action['id']}] {action['name']} (Source: {action['source']})")
                print(f"    {action['description']}")
                
                allocation = action['allocation']
                sorted_alloc = sorted(allocation.items(), key=lambda x: x[1], reverse=True)
                
                print(f"\n    Allocation (for ${budget/1000:.0f}K budget):")
                for ch, share in sorted_alloc:
                    if share > 0.01:
                        spend = share * budget
                        bar = "█" * int(share * 40)
                        print(f"    {ch:20s} ${spend:>8,.0f} ({share*100:5.1f}%) {bar}")


def test_revenue_sensitivity():
    """Test 7: Revenue sensitivity to allocation changes."""
    print_header("TEST 7: REVENUE SENSITIVITY ANALYSIS")
    print("How sensitive is revenue to allocation changes?")
    
    # Get base allocation
    base_data = get_recommendation(100000, 7, "ucb")
    
    if base_data.get('success'):
        base = base_data['recommendation']
        base_allocation = base['allocation']
        base_revenue = base['predicted_revenue']
        
        print(f"Base Strategy: {base['action_name']}")
        print(f"Base Revenue: {format_currency(base_revenue)}")
        
        print_subheader("MARGINAL ROI BY CHANNEL")
        
        # Get marginal ROI
        response = requests.post(f"{BASE_URL}/mmm/marginal-roi", 
                                json={"allocation": base_allocation})
        roi_data = response.json()
        
        if roi_data.get('success'):
            marginal = roi_data['marginal_rois']
            ranking = roi_data['ranking']
            
            headers = ["Rank", "Channel", "Marginal ROI", "Interpretation"]
            rows = []
            
            for i, ch in enumerate(ranking):
                roi = marginal[ch]
                if roi > 1:
                    interp = "🟢 Highly Profitable"
                elif roi > 0:
                    interp = "🟡 Profitable"
                elif roi > -1:
                    interp = "🟠 Low Return"
                else:
                    interp = "🔴 Negative Return"
                
                rows.append([i+1, ch, f"{roi:.4f}", interp])
            
            print(tabulate(rows, headers=headers, tablefmt="grid"))
            
            print(f"""
📈 INTERPRETATION:
   - Marginal ROI > 1: Each additional $1 returns more than $1 in revenue
   - Marginal ROI = 0.5: Each $1 returns $0.50 in revenue  
   - Marginal ROI < 0: Spending more actually decreases revenue (saturation)
   
💡 RECOMMENDATION:
   - Shift budget TO: {', '.join(ranking[:3])}
   - Shift budget FROM: {', '.join(ranking[-3:])}
""")


def run_all_tests():
    """Run all allocation tests."""
    print_header("MARKETING MIX RL - BUDGET ALLOCATION ANALYSIS")
    print(f"API: {BASE_URL}")
    
    # Check API
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.json().get('status') != 'healthy':
            print("❌ API not healthy")
            return
    except:
        print("❌ Cannot connect to API. Run: python main.py --api")
        return
    
    print("✅ API Connected\n")
    
    # Run tests
    test_varying_budgets()
    test_varying_months()
    test_promotion_impact()
    test_nps_impact()
    test_algorithm_comparison()
    test_all_strategies()
    test_revenue_sensitivity()
    
    print_header("ANALYSIS COMPLETE")


if __name__ == "__main__":
    # Check if tabulate is installed
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing tabulate...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'tabulate'])
        from tabulate import tabulate
    
    run_all_tests()