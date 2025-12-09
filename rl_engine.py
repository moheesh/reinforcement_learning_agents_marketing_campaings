"""
RL Engine
=========
Implements Q-Learning (Value-Based) and UCB Contextual Bandit (Exploration Strategy).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
import joblib
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TrainingHistory:
    """Store training metrics over episodes."""
    rewards: List[float] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    states: List[int] = field(default_factory=list)
    epsilon_values: List[float] = field(default_factory=list)
    q_value_changes: List[float] = field(default_factory=list)
    
    def add(self, reward: float, action: int, state: int, 
            epsilon: float = None, q_change: float = None):
        self.rewards.append(reward)
        self.actions.append(action)
        self.states.append(state)
        if epsilon is not None:
            self.epsilon_values.append(epsilon)
        if q_change is not None:
            self.q_value_changes.append(q_change)
    
    def get_summary(self, window: int = 1000) -> Dict:
        """Get summary statistics."""
        if len(self.rewards) == 0:
            return {}
        
        return {
            'total_episodes': len(self.rewards),
            'mean_reward': np.mean(self.rewards),
            'std_reward': np.std(self.rewards),
            'last_window_mean': np.mean(self.rewards[-window:]) if len(self.rewards) >= window else np.mean(self.rewards),
            'max_reward': np.max(self.rewards),
            'min_reward': np.min(self.rewards),
            'unique_actions': len(set(self.actions)),
            'unique_states': len(set(self.states))
        }


class QLearningAgent:
    """
    Tabular Q-Learning Agent (Value-Based RL).
    
    Learns Q(s, a) = expected reward for taking action a in state s.
    Uses epsilon-greedy exploration.
    """
    
    def __init__(self, n_states: int, n_actions: int, config: Dict):
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Hyperparameters
        self.alpha = config.get('alpha', 0.1)
        self.gamma = config.get('gamma', 0.0)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.9995)
        
        # Q-table: state x action -> value
        self.Q = np.zeros((n_states, n_actions))
        
        # Visit counts for analysis
        self.visit_counts = np.zeros((n_states, n_actions))
        
        # Training history
        self.history = TrainingHistory()
        self.trained = False
    
    def select_action(self, state: int, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.Q[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int = None) -> float:
        old_q = self.Q[state, action]
        
        if self.gamma > 0 and next_state is not None:
            target = reward + self.gamma * np.max(self.Q[next_state])
        else:
            target = reward
        
        self.Q[state, action] = old_q + self.alpha * (target - old_q)
        self.visit_counts[state, action] += 1
        
        return abs(self.Q[state, action] - old_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)
    
    def get_action_values(self, state: int) -> np.ndarray:
        return self.Q[state].copy()
    
    def get_best_action(self, state: int) -> Tuple[int, float]:
        action = np.argmax(self.Q[state])
        value = self.Q[state, action]
        return action, value
    
    def get_state_value(self, state: int) -> float:
        return np.max(self.Q[state])
    
    def save(self, filepath: str):
        state = {
            'Q': self.Q, 'visit_counts': self.visit_counts,
            'n_states': self.n_states, 'n_actions': self.n_actions,
            'alpha': self.alpha, 'gamma': self.gamma,
            'epsilon': self.epsilon, 'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'history': self.history, 'trained': self.trained
        }
        joblib.dump(state, filepath)
    
    def load(self, filepath: str):
        state = joblib.load(filepath)
        self.Q = state['Q']
        self.visit_counts = state['visit_counts']
        self.n_states = state['n_states']
        self.n_actions = state['n_actions']
        self.alpha = state['alpha']
        self.gamma = state['gamma']
        self.epsilon = state['epsilon']
        self.epsilon_end = state['epsilon_end']
        self.epsilon_decay = state['epsilon_decay']
        self.history = state['history']
        self.trained = state['trained']


class UCBAgent:
    """
    UCB Contextual Bandit Agent (Exploration Strategy).
    
    Uses Upper Confidence Bound for action selection:
    UCB(ctx, a) = Q(ctx, a) + c * sqrt(log(t) / N(ctx, a))
    """
    
    def __init__(self, n_contexts: int, n_actions: int, config: Dict):
        self.n_contexts = n_contexts
        self.n_actions = n_actions
        self.c = config.get('c', 1.5)
        self.Q = np.zeros((n_contexts, n_actions))
        self.N = np.zeros((n_contexts, n_actions))
        self.t = 0
        self.history = TrainingHistory()
        self.trained = False
    
    def select_action(self, context: int, explore: bool = True) -> int:
        self.t += 1
        
        unvisited = np.where(self.N[context] == 0)[0]
        if len(unvisited) > 0 and explore:
            return np.random.choice(unvisited)
        
        if explore:
            ucb_values = self._compute_ucb(context)
            return np.argmax(ucb_values)
        else:
            return np.argmax(self.Q[context])
    
    def _compute_ucb(self, context: int) -> np.ndarray:
        ucb_values = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            if self.N[context, a] == 0:
                ucb_values[a] = float('inf')
            else:
                exploitation = self.Q[context, a]
                exploration = self.c * np.sqrt(np.log(self.t + 1) / self.N[context, a])
                ucb_values[a] = exploitation + exploration
        
        return ucb_values
    
    def update(self, context: int, action: int, reward: float) -> float:
        old_q = self.Q[context, action]
        self.N[context, action] += 1
        n = self.N[context, action]
        self.Q[context, action] = old_q + (reward - old_q) / n
        return abs(self.Q[context, action] - old_q)
    
    def get_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)
    
    def get_action_values(self, context: int) -> np.ndarray:
        return self.Q[context].copy()
    
    def get_ucb_values(self, context: int) -> np.ndarray:
        return self._compute_ucb(context)
    
    def get_exploration_bonus(self, context: int) -> np.ndarray:
        bonuses = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            if self.N[context, a] > 0:
                bonuses[a] = self.c * np.sqrt(np.log(self.t + 1) / self.N[context, a])
            else:
                bonuses[a] = float('inf')
        return bonuses
    
    def get_best_action(self, context: int) -> Tuple[int, float]:
        action = np.argmax(self.Q[context])
        value = self.Q[context, action]
        return action, value
    
    def save(self, filepath: str):
        state = {
            'Q': self.Q, 'N': self.N, 't': self.t,
            'n_contexts': self.n_contexts, 'n_actions': self.n_actions,
            'c': self.c, 'history': self.history, 'trained': self.trained
        }
        joblib.dump(state, filepath)
    
    def load(self, filepath: str):
        state = joblib.load(filepath)
        self.Q = state['Q']
        self.N = state['N']
        self.t = state['t']
        self.n_contexts = state['n_contexts']
        self.n_actions = state['n_actions']
        self.c = state['c']
        self.history = state['history']
        self.trained = state['trained']


class RLEngine:
    """
    Main RL Engine that orchestrates training and recommendation.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_path = Path(self.config['paths']['models'])
        self.outputs_path = Path(self.config['paths']['outputs'])
        self.outputs_path.mkdir(exist_ok=True)
        self.rl_config = self.config['rl']
        
        self.q_agent: Optional[QLearningAgent] = None
        self.ucb_agent: Optional[UCBAgent] = None
        self.processor = None
        self.mmm = None
        
        self.reward_scale = float(self.rl_config['reward']['scale'])
        self.reward_objective = str(self.rl_config['reward']['objective'])
    
    def set_environment(self, processor, mmm):
        self.processor = processor
        self.mmm = mmm
    
    def compute_reward(self, action_id: int, budget: float, 
                       context: Dict[str, float] = None) -> float:
        spends = self.processor.get_allocation_for_action(action_id, budget)
        predicted_revenue = self.mmm.predict(spends, context)
        
        if hasattr(predicted_revenue, '__iter__'):
            predicted_revenue = float(predicted_revenue[0]) if len(predicted_revenue) > 0 else 0.0
        else:
            predicted_revenue = float(predicted_revenue)
        
        if self.reward_objective == 'revenue':
            reward = predicted_revenue / self.reward_scale
        elif self.reward_objective == 'roas':
            reward = (predicted_revenue - budget) / max(budget, 1)
        elif self.reward_objective == 'profit':
            reward = (predicted_revenue - budget) / self.reward_scale
        else:
            reward = predicted_revenue / self.reward_scale
        
        return float(reward)
    
    def train_q_learning(self, n_episodes: int = None, verbose: bool = True) -> Dict:
        if self.processor is None or self.mmm is None:
            raise ValueError("Set environment first with set_environment()")
        
        config = self.rl_config['q_learning']
        n_episodes = n_episodes or config['episodes']
        
        print(f"Training Q-Learning Agent...")
        print(f"  States: {self.processor.n_states}")
        print(f"  Actions: {self.processor.n_actions}")
        print(f"  Episodes: {n_episodes}")
        
        self.q_agent = QLearningAgent(
            n_states=self.processor.n_states,
            n_actions=self.processor.n_actions,
            config=config
        )
        
        df = self.processor.processed_data
        historical_states = [self.processor.get_state_index_from_row(row) 
                           for _, row in df.iterrows()]
        historical_budgets = df['total_spend'].values
        
        historical_contexts = []
        for _, row in df.iterrows():
            ctx = {
                'nps': row.get('nps', 50),
                'month': row.get('month', 6),
                'has_promotion': row.get('has_promotion', 0)
            }
            historical_contexts.append(ctx)
        
        for episode in range(n_episodes):
            idx = np.random.randint(len(df))
            state = historical_states[idx]
            budget = historical_budgets[idx]
            context = historical_contexts[idx]
            
            action = self.q_agent.select_action(state, explore=True)
            reward = self.compute_reward(action, budget, context)
            q_change = self.q_agent.update(state, action, reward)
            self.q_agent.decay_epsilon()
            
            self.q_agent.history.add(
                reward=reward, action=action, state=state,
                epsilon=self.q_agent.epsilon, q_change=q_change
            )
            
            if verbose and (episode + 1) % 5000 == 0:
                recent_rewards = self.q_agent.history.rewards[-1000:]
                print(f"  Episode {episode + 1}: "
                      f"Mean Reward = {np.mean(recent_rewards):.4f}, "
                      f"Epsilon = {self.q_agent.epsilon:.4f}")
        
        self.q_agent.trained = True
        self.q_agent.save(self.models_path / "q_agent.joblib")
        print(f"Q-Learning training complete. Saved to models/q_agent.joblib")
        
        return self.q_agent.history.get_summary()
    
    def train_ucb(self, n_episodes: int = None, verbose: bool = True) -> Dict:
        if self.processor is None or self.mmm is None:
            raise ValueError("Set environment first with set_environment()")
        
        config = self.rl_config['ucb']
        n_episodes = n_episodes or config['episodes']
        n_contexts = 12
        
        print(f"Training UCB Contextual Bandit Agent...")
        print(f"  Contexts: {n_contexts} (months)")
        print(f"  Actions: {self.processor.n_actions}")
        print(f"  Episodes: {n_episodes}")
        print(f"  Exploration c: {config['c']}")
        
        self.ucb_agent = UCBAgent(
            n_contexts=n_contexts,
            n_actions=self.processor.n_actions,
            config=config
        )
        
        df = self.processor.processed_data
        historical_months = df['month_index'].values
        historical_budgets = df['total_spend'].values
        
        historical_contexts = []
        for _, row in df.iterrows():
            ctx = {
                'nps': row.get('nps', 50),
                'month': row.get('month', 6),
                'has_promotion': row.get('has_promotion', 0)
            }
            historical_contexts.append(ctx)
        
        for episode in range(n_episodes):
            idx = np.random.randint(len(df))
            context_idx = int(historical_months[idx])
            budget = historical_budgets[idx]
            context = historical_contexts[idx]
            
            action = self.ucb_agent.select_action(context_idx, explore=True)
            reward = self.compute_reward(action, budget, context)
            q_change = self.ucb_agent.update(context_idx, action, reward)
            
            self.ucb_agent.history.add(
                reward=reward, action=action, state=context_idx, q_change=q_change
            )
            
            if verbose and (episode + 1) % 5000 == 0:
                recent_rewards = self.ucb_agent.history.rewards[-1000:]
                print(f"  Episode {episode + 1}: "
                      f"Mean Reward = {np.mean(recent_rewards):.4f}, "
                      f"Actions explored = {np.sum(self.ucb_agent.N > 0)}")
        
        self.ucb_agent.trained = True
        self.ucb_agent.save(self.models_path / "ucb_agent.joblib")
        print(f"UCB training complete. Saved to models/ucb_agent.joblib")
        
        return self.ucb_agent.history.get_summary()
    
    def _save_allocation_pie_chart(self, recommendation: Dict, save_path: Path = None):
        """Generate and save pie chart for budget allocation."""
        allocation = recommendation.get('allocation', {})
        if not allocation:
            return None
        
        filtered = {k: v for k, v in allocation.items() if v > 0}
        if not filtered:
            return None
        
        channels = list(filtered.keys())
        values = list(filtered.values())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(channels)))
        
        max_idx = values.index(max(values))
        explode = [0.05 if i == max_idx else 0.02 for i in range(len(channels))]
        
        def make_autopct(vals):
            def autopct(pct):
                total = sum(vals)
                val = pct / 100. * total
                if pct > 3:
                    return f'${val:,.0f}\n({pct:.1f}%)'
                return ''
            return autopct
        
        wedges, texts, autotexts = ax.pie(
            values, labels=channels, autopct=make_autopct(values),
            colors=colors, explode=explode, shadow=True, startangle=90
        )
        
        plt.setp(autotexts, size=9, weight='bold')
        plt.setp(texts, size=10)
        
        algo = recommendation.get('algorithm', 'RL').upper()
        budget = recommendation.get('budget', 0)
        month = recommendation.get('month', 0)
        revenue = recommendation.get('predicted_revenue', 0)
        roas = recommendation.get('predicted_roas', 0)
        action_name = recommendation.get('action_name', 'Unknown')
        
        title = f"{algo} Recommended Budget Allocation\n"
        title += f"Action: {action_name} | Budget: ${budget:,.0f} | Month: {month}\n"
        title += f"Predicted Revenue: ${revenue:,.0f} | ROAS: {roas:.2%}"
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        
        ax.legend(wedges, [f'{ch}: ${v:,.0f}' for ch, v in zip(channels, values)],
                  title="Channels", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.outputs_path / "allocation_recommendation.png"
        
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved allocation chart: {save_path}")
        return str(save_path)
    
    def recommend(self, budget: float, month: int, 
                  algorithm: str = 'q',
                  nps: float = 50,
                  has_promotion: int = 0,
                  constraints: Dict = None,
                  save_chart: bool = True) -> Dict:
        """
        Get budget allocation recommendation.
        
        Args:
            budget: Total budget to allocate
            month: Month (1-12)
            algorithm: 'q' for Q-learning, 'ucb' for UCB, 'compare' for both
            nps: NPS score
            has_promotion: Whether month has promotion
            constraints: Optional constraints like {'TV': {'min': 1000}}
            save_chart: If True, saves pie chart to outputs folder
        
        Returns:
            Recommendation dictionary
        """
        context = {
            'nps': float(nps),
            'month': int(month),
            'has_promotion': int(has_promotion)
        }
        
        if algorithm == 'compare':
            q_rec = self._get_recommendation('q', budget, month, context, constraints)
            ucb_rec = self._get_recommendation('ucb', budget, month, context, constraints)
            
            if save_chart:
                self._save_allocation_pie_chart(q_rec)
                self._save_allocation_pie_chart(ucb_rec)
            
            return {
                'q_learning': q_rec,
                'ucb': ucb_rec,
                'comparison': {
                    'revenue_diff': float(q_rec['predicted_revenue'] - ucb_rec['predicted_revenue']),
                    'same_action': q_rec['action_id'] == ucb_rec['action_id']
                }
            }
        else:
            rec = self._get_recommendation(algorithm, budget, month, context, constraints)
            
            if save_chart:
                self._save_allocation_pie_chart(rec)
            
            return rec
    
    def _get_recommendation(self, algorithm: str, budget: float, 
                            month: int, context: Dict,
                            constraints: Dict = None) -> Dict:
        month_index = month - 1
        
        if algorithm == 'q':
            if self.q_agent is None:
                self.q_agent = QLearningAgent(
                    self.processor.n_states, 
                    self.processor.n_actions,
                    self.rl_config['q_learning']
                )
                self.q_agent.load(self.models_path / "q_agent.joblib")
            
            state_values = {
                'month_index': month_index,
                'budget_level': budget,
                'nps_level': context['nps'],
                'revenue_trend': 0,
                'has_promotion': context['has_promotion']
            }
            state = self.processor.get_state_index(state_values)
            
            action, q_value = self.q_agent.get_best_action(state)
            action_values = self.q_agent.get_action_values(state)
            
        elif algorithm == 'ucb':
            if self.ucb_agent is None:
                self.ucb_agent = UCBAgent(
                    12, self.processor.n_actions,
                    self.rl_config['ucb']
                )
                self.ucb_agent.load(self.models_path / "ucb_agent.joblib")
            
            action, q_value = self.ucb_agent.get_best_action(month_index)
            action_values = self.ucb_agent.get_action_values(month_index)
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        action = int(action)
        q_value = float(q_value)
        
        if constraints:
            action = self._apply_constraints(action, budget, constraints, action_values)
        
        allocation = self.processor.get_allocation_for_action(action, budget)
        allocation = {k: float(v) for k, v in allocation.items()}
        
        predicted_revenue = self.mmm.predict(allocation, context)
        predicted_revenue = float(predicted_revenue)
        
        action_info = self.processor.actions[action]
        allocation_shares = {k: float(v) for k, v in action_info['allocation'].items()}
        
        return {
            'algorithm': algorithm,
            'budget': float(budget),
            'month': int(month),
            'action_id': int(action),
            'action_name': str(action_info['name']),
            'action_description': str(action_info['description']),
            'allocation': allocation,
            'allocation_shares': allocation_shares,
            'predicted_revenue': float(predicted_revenue),
            'predicted_roas': float((predicted_revenue - budget) / budget) if budget > 0 else 0.0,
            'q_value': float(q_value),
            'context': {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in context.items()}
        }
    
    def _apply_constraints(self, action: int, budget: float, 
                          constraints: Dict, action_values: np.ndarray) -> int:
        allocation = self.processor.get_allocation_for_action(action, budget)
        
        if self._check_constraints(allocation, constraints):
            return action
        
        sorted_actions = np.argsort(action_values)[::-1]
        
        for alt_action in sorted_actions:
            alt_allocation = self.processor.get_allocation_for_action(alt_action, budget)
            if self._check_constraints(alt_allocation, constraints):
                return alt_action
        
        return action
    
    def _check_constraints(self, allocation: Dict, constraints: Dict) -> bool:
        for channel, rules in constraints.items():
            value = allocation.get(channel, 0)
            
            if 'min' in rules and value < rules['min']:
                return False
            if 'max' in rules and value > rules['max']:
                return False
            if 'min_pct' in rules:
                total = sum(allocation.values())
                if value / total < rules['min_pct']:
                    return False
            if 'max_pct' in rules:
                total = sum(allocation.values())
                if value / total > rules['max_pct']:
                    return False
        
        return True
    
    def get_policy_summary(self, algorithm: str = 'q') -> Dict:
        if algorithm == 'q':
            if self.q_agent is None:
                return {'error': 'Q-agent not trained'}
            
            policy = self.q_agent.get_policy()
            visit_counts = self.q_agent.visit_counts
            action_counts = np.bincount(policy, minlength=self.processor.n_actions)
            
            return {
                'algorithm': 'q_learning',
                'n_states': int(self.q_agent.n_states),
                'n_actions': int(self.q_agent.n_actions),
                'action_distribution': {
                    self.processor.actions[i]['name']: int(count) 
                    for i, count in enumerate(action_counts)
                },
                'most_visited_state_action': [
                    int(x) for x in np.unravel_index(
                        np.argmax(visit_counts), visit_counts.shape
                    )
                ],
                'training_summary': {
                    'total_episodes': int(self.q_agent.history.get_summary().get('total_episodes', 0)),
                    'mean_reward': float(self.q_agent.history.get_summary().get('mean_reward', 0)),
                    'std_reward': float(self.q_agent.history.get_summary().get('std_reward', 0)),
                    'max_reward': float(self.q_agent.history.get_summary().get('max_reward', 0)),
                    'min_reward': float(self.q_agent.history.get_summary().get('min_reward', 0)),
                    'unique_actions': int(self.q_agent.history.get_summary().get('unique_actions', 0)),
                    'unique_states': int(self.q_agent.history.get_summary().get('unique_states', 0))
                }
            }
        
        elif algorithm == 'ucb':
            if self.ucb_agent is None:
                return {'error': 'UCB agent not trained'}
            
            policy = self.ucb_agent.get_policy()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            policy_by_month = {
                month_names[i]: str(self.processor.actions[int(policy[i])]['name'])
                for i in range(len(policy))
            }
            
            return {
                'algorithm': 'ucb',
                'n_contexts': int(self.ucb_agent.n_contexts),
                'n_actions': int(self.ucb_agent.n_actions),
                'exploration_c': float(self.ucb_agent.c),
                'total_steps': int(self.ucb_agent.t),
                'policy_by_month': policy_by_month,
                'training_summary': {
                    'total_episodes': int(self.ucb_agent.history.get_summary().get('total_episodes', 0)),
                    'mean_reward': float(self.ucb_agent.history.get_summary().get('mean_reward', 0)),
                    'std_reward': float(self.ucb_agent.history.get_summary().get('std_reward', 0)),
                    'max_reward': float(self.ucb_agent.history.get_summary().get('max_reward', 0)),
                    'min_reward': float(self.ucb_agent.history.get_summary().get('min_reward', 0)),
                    'unique_actions': int(self.ucb_agent.history.get_summary().get('unique_actions', 0)),
                    'unique_states': int(self.ucb_agent.history.get_summary().get('unique_states', 0))
                }
            }
        
        else:
            return {'error': f'Unknown algorithm: {algorithm}'}
    
    def load_agents(self):
        q_path = self.models_path / "q_agent.joblib"
        ucb_path = self.models_path / "ucb_agent.joblib"
        
        if q_path.exists():
            self.q_agent = QLearningAgent(
                self.processor.n_states,
                self.processor.n_actions,
                self.rl_config['q_learning']
            )
            self.q_agent.load(q_path)
            print(f"Loaded Q-agent from {q_path}")
        
        if ucb_path.exists():
            self.ucb_agent = UCBAgent(
                12, self.processor.n_actions,
                self.rl_config['ucb']
            )
            self.ucb_agent.load(ucb_path)
            print(f"Loaded UCB agent from {ucb_path}")


if __name__ == "__main__":
    from data_processor import DataProcessor
    from mmm_model import MMMModel
    import json
    
    processor = DataProcessor("config.yaml")
    processor.load_data()
    processor.load_special_sales()
    processor.engineer_features()
    processor.build_state_space()
    processor.build_action_space()
    
    mmm = MMMModel("config.yaml")
    mmm.train(processor.processed_data)
    
    processor.add_roi_proportional_action(mmm.channel_rois)
    processor.save()
    
    rl = RLEngine("config.yaml")
    rl.set_environment(processor, mmm)
    
    print("\n" + "="*60)
    print("TRAINING Q-LEARNING")
    print("="*60)
    q_summary = rl.train_q_learning(n_episodes=30000)
    print(json.dumps(q_summary, indent=2))
    
    print("\n" + "="*60)
    print("TRAINING UCB")
    print("="*60)
    ucb_summary = rl.train_ucb(n_episodes=30000)
    print(json.dumps(ucb_summary, indent=2))
    
    print("\n" + "="*60)
    print("TEST RECOMMENDATIONS")
    print("="*60)
    
    for budget in [50, 100, 200]:
        print(f"\n--- Budget: {budget} ---")
        rec = rl.recommend(
            budget=budget,
            month=7,
            algorithm='compare',
            nps=50,
            has_promotion=1,
            save_chart=True
        )
        
        print(f"Q-Learning: {rec['q_learning']['action_name']}")
        print(f"  Predicted Revenue: {rec['q_learning']['predicted_revenue']:,.0f}")
        print(f"UCB: {rec['ucb']['action_name']}")
        print(f"  Predicted Revenue: {rec['ucb']['predicted_revenue']:,.0f}")
    
    print("\n" + "="*60)
    print("POLICY SUMMARIES")
    print("="*60)
    
    print("\nQ-Learning Policy:")
    print(json.dumps(rl.get_policy_summary('q'), indent=2, default=str))
    
    print("\nUCB Policy:")
    print(json.dumps(rl.get_policy_summary('ucb'), indent=2, default=str))