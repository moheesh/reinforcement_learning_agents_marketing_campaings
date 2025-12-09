"""
Data Processor
==============
Loads raw data, engineers features, builds state space and action space.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yaml
import joblib
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Load, process, and feature-engineer marketing data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_path = Path(self.config['paths']['raw_data'])
        self.models_path = Path(self.config['paths']['models'])
        self.outputs_path = Path(self.config['paths']['outputs'])
        
        # Create directories
        self.models_path.mkdir(exist_ok=True)
        self.outputs_path.mkdir(exist_ok=True)
        
        self.channels: List[str] = self.config['channels']
        self.target: str = self.config['target']
        
        # Data containers
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.special_sales: Optional[pd.DataFrame] = None
        
        # State space info
        self.state_config: Dict = {}
        self.state_bins: Dict = {}
        self.n_states: int = 0
        
        # Action space info
        self.actions: List[Dict] = []
        self.n_actions: int = 0
        self.allocation_templates: np.ndarray = None
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_data(self) -> pd.DataFrame:
        """Load main data file."""
        filepath = self.raw_path / self.config['files']['main_data']
        print(f"Loading data from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        # Clean column names
        if df.columns[0] == '' or 'Unnamed' in str(df.columns[0]):
            df = df.drop(df.columns[0], axis=1)
        
        # Parse date
        date_col = None
        for col in ['Date', 'date', 'month', 'Month']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df['Date'] = pd.to_datetime(df[date_col], format='mixed')
            if date_col != 'Date':
                df = df.drop(columns=[date_col])
        
        # Ensure numeric for channels and target
        for col in self.channels + [self.target]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        self.raw_data = df
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Channels found: {[c for c in self.channels if c in df.columns]}")
        
        return df
    
    def load_special_sales(self) -> Optional[pd.DataFrame]:
        """Load special sales/promotion calendar."""
        filepath = self.raw_path / self.config['files']['special_sales']
        
        if not filepath.exists():
            print(f"Special sales file not found at {filepath}, skipping...")
            return None
        
        print(f"Loading special sales from {filepath}...")
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        
        self.special_sales = df
        print(f"Loaded {len(df)} special sale dates")
        
        return df
    
    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    
    def engineer_features(self) -> pd.DataFrame:
        """Create all features from raw data."""
        if self.raw_data is None:
            raise ValueError("Load data first")
        
        df = self.raw_data.copy()
        print("Engineering features...")
        
        # --- Temporal Features ---
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['month_index'] = df['month'] - 1  # 0-11
        df['quarter_index'] = df['quarter'] - 1  # 0-3
        df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
        df['day_of_year'] = df['Date'].dt.dayofyear
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # --- Budget Features ---
        available_channels = [c for c in self.channels if c in df.columns]
        df['total_spend'] = df[available_channels].sum(axis=1)
        
        # Spend shares (allocation percentages)
        for ch in available_channels:
            df[f'{ch}_share'] = df[ch] / df['total_spend'].replace(0, 1)
        
        # Budget change (MoM)
        df['budget_change'] = df['total_spend'].pct_change().fillna(0)
        
        # --- Performance Features ---
        # Revenue trend (3-month rolling slope)
        df['revenue_ma3'] = df[self.target].rolling(3, min_periods=1).mean()
        df['revenue_trend'] = df[self.target].pct_change(periods=3).fillna(0)
        
        # ROAS
        df['roas'] = df[self.target] / df['total_spend'].replace(0, 1)
        
        # YoY growth (if enough data)
        if len(df) > 12:
            df['yoy_growth'] = df[self.target].pct_change(periods=12).fillna(0)
        else:
            df['yoy_growth'] = 0
        
        # --- NPS Features ---
        if 'NPS' in df.columns:
            df['nps'] = df['NPS']
        else:
            df['nps'] = 50  # Default if not available
        
        # --- Promotion Features ---
        df['has_promotion'] = 0
        if self.special_sales is not None:
            promo_months = self.special_sales['Date'].dt.to_period('M').unique()
            df['promo_period'] = df['Date'].dt.to_period('M')
            df['has_promotion'] = df['promo_period'].isin(promo_months).astype(int)
            df = df.drop(columns=['promo_period'])
        
        # --- Lag Features ---
        for ch in available_channels[:3]:  # Top 3 channels
            df[f'{ch}_lag1'] = df[ch].shift(1).fillna(0)
        
        df['revenue_lag1'] = df[self.target].shift(1).fillna(df[self.target].mean())
        
        self.processed_data = df
        print(f"Created {len(df.columns)} features")
        
        return df
    
    # =========================================================================
    # STATE SPACE
    # =========================================================================
    
    def build_state_space(self) -> Dict:
        """Build state space configuration from config."""
        if self.processed_data is None:
            raise ValueError("Engineer features first")
        
        df = self.processed_data
        state_cfg = self.config['state']
        
        print("Building state space...")
        
        active_dims = []
        total_states = 1
        
        for dim_name, dim_config in state_cfg.items():
            if not dim_config.get('enabled', False):
                continue
            
            bins = dim_config['bins']
            
            # Determine bin edges based on method
            if dim_name == 'month_index':
                edges = np.arange(bins + 1)  # 0-12 for months
                
            elif dim_name == 'quarter':
                edges = np.arange(bins + 1)  # 0-4 for quarters
                
            elif dim_name == 'has_promotion':
                edges = np.array([0, 1, 2])  # Binary
                
            elif 'thresholds' in dim_config:
                thresholds = dim_config['thresholds']
                edges = np.array([-np.inf] + thresholds + [np.inf])
                
            elif dim_config.get('method') == 'quantile':
                source_col = self._get_source_column(dim_name)
                if source_col in df.columns:
                    edges = np.percentile(df[source_col].dropna(), 
                                         np.linspace(0, 100, bins + 1))
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                else:
                    edges = np.linspace(0, 1, bins + 1)
            else:
                source_col = self._get_source_column(dim_name)
                if source_col in df.columns:
                    edges = np.linspace(df[source_col].min(), 
                                       df[source_col].max(), bins + 1)
                else:
                    edges = np.linspace(0, 1, bins + 1)
            
            self.state_bins[dim_name] = {
                'bins': bins,
                'edges': edges,
                'source': self._get_source_column(dim_name)
            }
            
            active_dims.append(dim_name)
            total_states *= bins
            print(f"  {dim_name}: {bins} bins")
        
        self.state_config = {
            'dimensions': active_dims,
            'total_states': total_states,
            'bins': self.state_bins
        }
        
        self.n_states = total_states
        print(f"Total state space: {total_states} states")
        
        return self.state_config
    
    def _get_source_column(self, dim_name: str) -> str:
        """Map dimension name to data column."""
        mapping = {
            'month_index': 'month_index',
            'quarter': 'quarter_index',
            'budget_level': 'total_spend',
            'budget_change': 'budget_change',
            'nps_level': 'nps',
            'revenue_trend': 'revenue_trend',
            'roas_level': 'roas',
            'has_promotion': 'has_promotion',
            'week_of_year': 'week_of_year'
        }
        return mapping.get(dim_name, dim_name)
    
    def get_state_index(self, state_values: Dict[str, float]) -> int:
        """Convert state values to single state index."""
        index = 0
        multiplier = 1
        
        for dim_name in reversed(self.state_config['dimensions']):
            bin_info = self.state_bins[dim_name]
            value = state_values.get(dim_name, 0)
            
            # Discretize value to bin
            bin_idx = np.digitize(value, bin_info['edges']) - 1
            bin_idx = np.clip(bin_idx, 0, bin_info['bins'] - 1)
            
            index += bin_idx * multiplier
            multiplier *= bin_info['bins']
        
        return int(index)
    
    def get_state_from_row(self, row: pd.Series) -> Dict[str, float]:
        """Extract state values from a data row."""
        state = {}
        for dim_name in self.state_config['dimensions']:
            source_col = self.state_bins[dim_name]['source']
            state[dim_name] = row.get(source_col, 0)
        return state
    
    def get_state_index_from_row(self, row: pd.Series) -> int:
        """Get state index directly from data row."""
        state_values = self.get_state_from_row(row)
        return self.get_state_index(state_values)
    
    # =========================================================================
    # ACTION SPACE
    # =========================================================================
    
    def build_action_space(self) -> List[Dict]:
        """Discover actions from data + generate additional templates."""
        if self.processed_data is None:
            raise ValueError("Engineer features first")
        
        df = self.processed_data
        action_cfg = self.config['actions']
        available_channels = [c for c in self.channels if c in df.columns]
        n_channels = len(available_channels)
        
        print("Building action space...")
        
        actions = []
        
        # --- 1. Cluster Historical Allocations ---
        share_cols = [f'{ch}_share' for ch in available_channels]
        allocation_data = df[share_cols].values
        
        # Remove any rows with NaN
        valid_mask = ~np.isnan(allocation_data).any(axis=1)
        allocation_data = allocation_data[valid_mask]
        
        if len(allocation_data) > action_cfg['n_clusters']:
            print(f"  Clustering {len(allocation_data)} historical allocations...")
            
            kmeans = KMeans(
                n_clusters=action_cfg['n_clusters'],
                random_state=42,
                n_init=10
            )
            kmeans.fit(allocation_data)
            
            for i, centroid in enumerate(kmeans.cluster_centers_):
                # Normalize to sum to 1
                centroid = np.clip(centroid, 0, 1)
                centroid = centroid / centroid.sum()
                
                # Find dominant channel
                dominant_idx = np.argmax(centroid)
                dominant_channel = available_channels[dominant_idx]
                dominant_pct = centroid[dominant_idx] * 100
                
                actions.append({
                    'id': len(actions),
                    'name': f'Cluster_{i}_{dominant_channel[:3]}_{dominant_pct:.0f}pct',
                    'source': 'cluster',
                    'allocation': dict(zip(available_channels, centroid)),
                    'description': f'Historical pattern: {dominant_channel} dominant ({dominant_pct:.1f}%)'
                })
            
            print(f"  Found {action_cfg['n_clusters']} cluster-based actions")
        
        # --- 2. Balanced Allocation ---
        if action_cfg.get('add_balanced', True):
            balanced = np.ones(n_channels) / n_channels
            actions.append({
                'id': len(actions),
                'name': 'Balanced',
                'source': 'generated',
                'allocation': dict(zip(available_channels, balanced)),
                'description': 'Equal distribution across all channels'
            })
            print("  Added balanced action")
        
        # --- 3. Top-N Concentrated ---
        if action_cfg.get('add_top_n_concentrated'):
            # Use mean spend as proxy for importance (will be replaced by ROI after MMM)
            mean_spend = df[available_channels].mean()
            channel_rank = mean_spend.sort_values(ascending=False).index.tolist()
            
            for n in action_cfg['add_top_n_concentrated']:
                if n > n_channels:
                    continue
                
                alloc = np.zeros(n_channels)
                top_channels = channel_rank[:n]
                for ch in top_channels:
                    idx = available_channels.index(ch)
                    alloc[idx] = 1.0 / n
                
                actions.append({
                    'id': len(actions),
                    'name': f'Top{n}_Concentrated',
                    'source': 'generated',
                    'allocation': dict(zip(available_channels, alloc)),
                    'description': f'Concentrate on top {n} channels: {", ".join(top_channels)}'
                })
            
            print(f"  Added {len(action_cfg['add_top_n_concentrated'])} concentrated actions")
        
        # --- Store Results ---
        self.actions = actions
        self.n_actions = len(actions)
        
        # Create allocation matrix for fast lookup
        self.allocation_templates = np.array([
            [a['allocation'].get(ch, 0) for ch in available_channels]
            for a in actions
        ])
        
        print(f"Total actions: {self.n_actions}")
        
        return actions
    
    def add_roi_proportional_action(self, channel_rois: Dict[str, float]):
        """Add action proportional to channel ROIs (call after MMM training)."""
        available_channels = [c for c in self.channels if c in self.processed_data.columns]
        
        rois = np.array([max(channel_rois.get(ch, 0), 0) for ch in available_channels])
        
        if rois.sum() > 0:
            alloc = rois / rois.sum()
        else:
            alloc = np.ones(len(available_channels)) / len(available_channels)
        
        action = {
            'id': len(self.actions),
            'name': 'ROI_Proportional',
            'source': 'mmm',
            'allocation': dict(zip(available_channels, alloc)),
            'description': 'Allocation proportional to MMM-estimated channel ROI'
        }
        
        self.actions.append(action)
        self.n_actions = len(self.actions)
        
        # Update allocation matrix
        self.allocation_templates = np.vstack([
            self.allocation_templates,
            alloc
        ])
        
        print(f"Added ROI-proportional action (total: {self.n_actions})")
    
    def get_allocation_for_action(self, action_id: int, budget: float) -> Dict[str, float]:
        """Convert action ID to actual spend amounts."""
        if action_id >= self.n_actions:
            raise ValueError(f"Invalid action {action_id}, max is {self.n_actions - 1}")
        
        allocation_shares = self.allocation_templates[action_id]
        available_channels = [c for c in self.channels if c in self.processed_data.columns]
        
        spends = {ch: share * budget for ch, share in zip(available_channels, allocation_shares)}
        return spends
    
    # =========================================================================
    # SAVE / LOAD
    # =========================================================================
    
    def save(self, filename: str = "data_processor.joblib"):
        """Save processor state."""
        state = {
            'channels': self.channels,
            'target': self.target,
            'state_config': self.state_config,
            'state_bins': self.state_bins,
            'n_states': self.n_states,
            'actions': self.actions,
            'n_actions': self.n_actions,
            'allocation_templates': self.allocation_templates
        }
        
        filepath = self.models_path / filename
        joblib.dump(state, filepath)
        print(f"Saved processor to {filepath}")
    
    def load(self, filename: str = "data_processor.joblib"):
        """Load processor state."""
        filepath = self.models_path / filename
        state = joblib.load(filepath)
        
        self.channels = state['channels']
        self.target = state['target']
        self.state_config = state['state_config']
        self.state_bins = state['state_bins']
        self.n_states = state['n_states']
        self.actions = state['actions']
        self.n_actions = state['n_actions']
        self.allocation_templates = state['allocation_templates']
        
        print(f"Loaded processor from {filepath}")
    
    def get_summary(self) -> Dict:
        """Return summary of processed data."""
        summary = {
            'n_rows': len(self.processed_data) if self.processed_data is not None else 0,
            'date_range': None,
            'channels': self.channels,
            'n_states': self.n_states,
            'state_dimensions': self.state_config.get('dimensions', []),
            'n_actions': self.n_actions,
            'action_names': [a['name'] for a in self.actions]
        }
        
        if self.processed_data is not None:
            summary['date_range'] = {
                'start': str(self.processed_data['Date'].min()),
                'end': str(self.processed_data['Date'].max())
            }
        
        return summary


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test the processor
    processor = DataProcessor("config.yaml")
    
    # Load data
    processor.load_data()
    processor.load_special_sales()
    
    # Engineer features
    processor.engineer_features()
    
    # Build state space
    processor.build_state_space()
    
    # Build action space
    processor.build_action_space()
    
    # Save
    processor.save()
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    import json
    print(json.dumps(processor.get_summary(), indent=2))