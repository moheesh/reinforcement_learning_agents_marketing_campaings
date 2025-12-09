"""
Marketing Mix Model (MMM)
=========================
Trains regression model to estimate channel effectiveness and predict revenue.
Includes adstock transformation and saturation curves.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from scipy.optimize import minimize_scalar
import yaml
import joblib
import warnings
warnings.filterwarnings('ignore')


class MMMModel:
    """Marketing Mix Model with adstock and saturation transforms."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_path = Path(self.config['paths']['models'])
        self.mmm_config = self.config['mmm']
        
        self.channels: List[str] = self.config['channels']
        self.target: str = self.config['target']
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        
        # Learned parameters
        self.adstock_decays: Dict[str, float] = {}
        self.saturation_params: Dict[str, float] = {}
        self.coefficients: Dict[str, float] = {}
        self.channel_contributions: Dict[str, float] = {}
        self.channel_rois: Dict[str, float] = {}
        
        # Metrics
        self.metrics: Dict[str, float] = {}
        
        # Store training data stats for prediction
        self.train_stats: Dict = {}
    
    # =========================================================================
    # TRANSFORMATIONS
    # =========================================================================
    
    def apply_adstock(self, x: np.ndarray, decay: float) -> np.ndarray:
        """
        Apply adstock transformation (carryover effect).
        adstock[t] = x[t] + decay * adstock[t-1]
        """
        adstocked = np.zeros_like(x, dtype=float)
        adstocked[0] = x[0]
        
        for t in range(1, len(x)):
            adstocked[t] = x[t] + decay * adstocked[t - 1]
        
        return adstocked
    
    def apply_saturation_hill(self, x: np.ndarray, half_saturation: float = None) -> np.ndarray:
        """
        Apply Hill saturation (diminishing returns).
        saturated = x / (x + half_saturation)
        """
        if half_saturation is None:
            half_saturation = np.median(x[x > 0]) if (x > 0).any() else 1
        
        # Avoid division by zero
        return x / (x + half_saturation + 1e-10)
    
    def apply_saturation_exp(self, x: np.ndarray, lambd: float = 0.001) -> np.ndarray:
        """
        Apply exponential saturation.
        saturated = 1 - exp(-lambda * x)
        """
        return 1 - np.exp(-lambd * x)
    
    def find_optimal_adstock(self, x: np.ndarray, y: np.ndarray) -> float:
        """Find optimal adstock decay via grid search."""
        decay_range = self.mmm_config['adstock']['decay_range']
        decay_steps = self.mmm_config['adstock']['decay_steps']
        
        best_decay = 0.0
        best_corr = -1
        
        for decay in np.linspace(decay_range[0], decay_range[1], decay_steps):
            adstocked = self.apply_adstock(x, decay)
            corr = np.corrcoef(adstocked, y)[0, 1]
            
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_decay = decay
        
        return best_decay
    
    # =========================================================================
    # FEATURE ENGINEERING FOR MMM
    # =========================================================================
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features with adstock and saturation transforms."""
        available_channels = [c for c in self.channels if c in df.columns]
        y = df[self.target].values
        
        features = []
        feature_names = []
        
        # --- Channel Features with Transforms ---
        for ch in available_channels:
            x = df[ch].values.astype(float)
            
            # Find optimal adstock decay
            if fit:
                decay = self.find_optimal_adstock(x, y)
                self.adstock_decays[ch] = decay
            else:
                decay = self.adstock_decays.get(ch, 0.5)
            
            # Apply adstock
            if self.mmm_config['adstock']['enabled']:
                x_transformed = self.apply_adstock(x, decay)
            else:
                x_transformed = x
            
            # Apply saturation
            if self.mmm_config['saturation']['enabled']:
                if self.mmm_config['saturation']['method'] == 'hill':
                    half_sat = np.median(x_transformed[x_transformed > 0]) if (x_transformed > 0).any() else 1
                    if fit:
                        self.saturation_params[ch] = half_sat
                    x_transformed = self.apply_saturation_hill(x_transformed, half_sat)
                else:
                    x_transformed = self.apply_saturation_exp(x_transformed)
            
            features.append(x_transformed)
            feature_names.append(ch)
        
        # --- Control Variables ---
        # NPS
        if 'nps' in df.columns:
            features.append(df['nps'].values)
            feature_names.append('nps')
        elif 'NPS' in df.columns:
            features.append(df['NPS'].values)
            feature_names.append('NPS')
        
        # Promotion indicator
        if 'has_promotion' in df.columns:
            features.append(df['has_promotion'].values)
            feature_names.append('has_promotion')
        
        # Seasonality (month sin/cos)
        if 'month_sin' in df.columns:
            features.append(df['month_sin'].values)
            features.append(df['month_cos'].values)
            feature_names.extend(['month_sin', 'month_cos'])
        
        # Trend
        if 'day_of_year' in df.columns:
            features.append(df['day_of_year'].values / 365)  # Normalize
            feature_names.append('trend')
        
        X = np.column_stack(features)
        
        if fit:
            self.feature_names = feature_names
        
        return X, y
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the MMM model."""
        print("Training Marketing Mix Model...")
        
        available_channels = [c for c in self.channels if c in df.columns]
        print(f"  Channels: {available_channels}")
        
        # Prepare features
        X, y = self.prepare_features(df, fit=True)
        print(f"  Features: {self.feature_names}")
        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        
        # Store training stats for prediction
        self.train_stats = {
            'y_mean': y.mean(),
            'y_std': y.std(),
            'X_mean': X.mean(axis=0),
            'X_std': X.std(axis=0)
        }
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        model_type = self.mmm_config['model_type']
        alpha = self.mmm_config['alpha']
        
        if model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        else:
            self.model = ElasticNet(alpha=alpha, l1_ratio=0.5)
        
        self.model.fit(X_scaled, y)
        
        # Calculate metrics
        y_pred = self.model.predict(X_scaled)
        
        self.metrics = {
            'r2': r2_score(y, y_pred),
            'mape': mean_absolute_percentage_error(y, y_pred) * 100,
            'rmse': np.sqrt(np.mean((y - y_pred) ** 2)),
            'n_samples': len(y)
        }
        
        # Cross-validation (if enough data)
        if len(y) >= 5:
            cv = min(5, len(y))
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='r2')
            self.metrics['cv_r2_mean'] = cv_scores.mean()
            self.metrics['cv_r2_std'] = cv_scores.std()
        
        print(f"  R²: {self.metrics['r2']:.4f}")
        print(f"  MAPE: {self.metrics['mape']:.2f}%")
        
        # Extract coefficients
        self._extract_coefficients(df, y)
        
        return self.metrics
    
    def _extract_coefficients(self, df: pd.DataFrame, y: np.ndarray):
        """Extract and interpret model coefficients."""
        available_channels = [c for c in self.channels if c in df.columns]
        
        # Raw coefficients (scaled)
        for i, feat in enumerate(self.feature_names):
            self.coefficients[feat] = self.model.coef_[i]
        
        # Channel contributions (% of predicted revenue)
        X, _ = self.prepare_features(df, fit=False)
        X_scaled = self.scaler.transform(X)
        
        base_pred = self.model.intercept_
        total_contribution = 0
        
        for i, feat in enumerate(self.feature_names):
            contrib = self.model.coef_[i] * X_scaled[:, i].mean()
            if feat in available_channels:
                self.channel_contributions[feat] = contrib
                total_contribution += abs(contrib)
        
        # Normalize contributions to percentages
        if total_contribution > 0:
            for ch in available_channels:
                if ch in self.channel_contributions:
                    self.channel_contributions[ch] = (
                        abs(self.channel_contributions[ch]) / total_contribution * 100
                    )
        
        # Calculate ROI (coefficient * scale factor)
        # ROI = marginal revenue per unit spend
        for ch in available_channels:
            if ch in self.coefficients:
                # Approximate ROI: positive coefficient = positive ROI
                coef = self.coefficients[ch]
                mean_spend = df[ch].mean()
                mean_revenue = y.mean()
                
                if mean_spend > 0:
                    # Rough ROI estimate
                    self.channel_rois[ch] = coef * (mean_revenue / mean_spend) / 100
                else:
                    self.channel_rois[ch] = 0
        
        print("\n  Channel Contributions:")
        for ch in available_channels:
            contrib = self.channel_contributions.get(ch, 0)
            roi = self.channel_rois.get(ch, 0)
            print(f"    {ch}: {contrib:.1f}% contribution, ROI={roi:.2f}")
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    def predict(self, spend_dict: Dict[str, float], context: Dict[str, float] = None) -> float:
        """
        Predict revenue for a given spend allocation.
        
        Args:
            spend_dict: {channel: spend_amount}
            context: {nps, has_promotion, month, etc.}
        
        Returns:
            Predicted revenue
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        available_channels = [c for c in self.channels if c in self.feature_names]
        
        # Build feature vector
        features = []
        
        for feat in self.feature_names:
            if feat in available_channels:
                # Channel spend - apply transforms
                x = spend_dict.get(feat, 0)
                
                # Apply adstock (single value, so just return as-is)
                decay = self.adstock_decays.get(feat, 0.5)
                x_transformed = x  # For single prediction, adstock doesn't change value
                
                # Apply saturation
                if self.mmm_config['saturation']['enabled']:
                    half_sat = self.saturation_params.get(feat, 1)
                    x_transformed = x_transformed / (x_transformed + half_sat + 1e-10)
                
                features.append(x_transformed)
            
            elif feat == 'nps' or feat == 'NPS':
                features.append(context.get('nps', 50) if context else 50)
            
            elif feat == 'has_promotion':
                features.append(context.get('has_promotion', 0) if context else 0)
            
            elif feat == 'month_sin':
                month = context.get('month', 6) if context else 6
                features.append(np.sin(2 * np.pi * month / 12))
            
            elif feat == 'month_cos':
                month = context.get('month', 6) if context else 6
                features.append(np.cos(2 * np.pi * month / 12))
            
            elif feat == 'trend':
                features.append(0.5)  # Mid-year default
            
            else:
                features.append(0)
        
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        
        # Ensure non-negative
        return max(0, prediction)
    
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Predict for entire dataframe."""
        X, _ = self.prepare_features(df, fit=False)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    def get_marginal_roi(self, base_spend: Dict[str, float], 
                         context: Dict[str, float] = None,
                         delta_pct: float = 0.01) -> Dict[str, float]:
        """
        Calculate marginal ROI for each channel.
        How much additional revenue per additional $1 spend.
        """
        base_revenue = self.predict(base_spend, context)
        marginal_rois = {}
        
        available_channels = [c for c in self.channels if c in self.feature_names]
        
        for ch in available_channels:
            # Increase spend by delta_pct
            modified_spend = base_spend.copy()
            original = modified_spend.get(ch, 0)
            delta = max(original * delta_pct, 1)  # At least $1
            modified_spend[ch] = original + delta
            
            new_revenue = self.predict(modified_spend, context)
            
            # Marginal ROI = change in revenue / change in spend
            marginal_rois[ch] = (new_revenue - base_revenue) / delta
        
        return marginal_rois
    
    def simulate_reallocation(self, 
                              current_spend: Dict[str, float],
                              from_channel: str,
                              to_channel: str,
                              amount: float,
                              context: Dict[str, float] = None) -> Dict:
        """
        Simulate moving budget from one channel to another.
        """
        current_revenue = self.predict(current_spend, context)
        
        new_spend = current_spend.copy()
        new_spend[from_channel] = max(0, new_spend.get(from_channel, 0) - amount)
        new_spend[to_channel] = new_spend.get(to_channel, 0) + amount
        
        new_revenue = self.predict(new_spend, context)
        
        return {
            'current_revenue': current_revenue,
            'new_revenue': new_revenue,
            'revenue_change': new_revenue - current_revenue,
            'revenue_change_pct': (new_revenue - current_revenue) / current_revenue * 100,
            'from_channel': from_channel,
            'to_channel': to_channel,
            'amount_moved': amount
        }
    
    # =========================================================================
    # SAVE / LOAD
    # =========================================================================
    
    def save(self, filename: str = "mmm_model.joblib"):
        """Save model and all parameters."""
        state = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'adstock_decays': self.adstock_decays,
            'saturation_params': self.saturation_params,
            'coefficients': self.coefficients,
            'channel_contributions': self.channel_contributions,
            'channel_rois': self.channel_rois,
            'metrics': self.metrics,
            'train_stats': self.train_stats,
            'config': self.mmm_config
        }
        
        filepath = self.models_path / filename
        joblib.dump(state, filepath)
        print(f"Saved MMM to {filepath}")
    
    def load(self, filename: str = "mmm_model.joblib"):
        """Load model and all parameters."""
        filepath = self.models_path / filename
        state = joblib.load(filepath)
        
        self.model = state['model']
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.adstock_decays = state['adstock_decays']
        self.saturation_params = state['saturation_params']
        self.coefficients = state['coefficients']
        self.channel_contributions = state['channel_contributions']
        self.channel_rois = state['channel_rois']
        self.metrics = state['metrics']
        self.train_stats = state['train_stats']
        
        print(f"Loaded MMM from {filepath}")
    
    def get_summary(self) -> Dict:
        """Return model summary."""
        return {
            'metrics': self.metrics,
            'adstock_decays': self.adstock_decays,
            'channel_contributions': self.channel_contributions,
            'channel_rois': self.channel_rois,
            'feature_names': self.feature_names
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from data_processor import DataProcessor
    
    # Load and process data
    processor = DataProcessor("config.yaml")
    processor.load_data()
    processor.load_special_sales()
    processor.engineer_features()
    
    # Train MMM
    mmm = MMMModel("config.yaml")
    metrics = mmm.train(processor.processed_data)
    
    # Save model
    mmm.save()
    
    # Test prediction
    print("\n" + "="*50)
    print("TEST PREDICTION")
    print("="*50)
    
    test_spend = {
        'TV': 50,
        'Digital': 30,
        'Sponsorship': 200,
        'Content.Marketing': 10,
        'Online.marketing': 100,
        'Affiliates': 50,
        'SEM': 50,
        'Radio': 10,
        'Other': 100
    }
    
    context = {'nps': 50, 'month': 7, 'has_promotion': 1}
    
    predicted_revenue = mmm.predict(test_spend, context)
    print(f"Test spend: {sum(test_spend.values())}")
    print(f"Predicted revenue: {predicted_revenue:,.0f}")
    
    # Marginal ROI
    print("\n" + "="*50)
    print("MARGINAL ROI")
    print("="*50)
    
    marginal = mmm.get_marginal_roi(test_spend, context)
    for ch, roi in sorted(marginal.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ch}: {roi:.4f}")
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    import json
    print(json.dumps(mmm.get_summary(), indent=2, default=str))