#!/usr/bin/env python3
"""
Model Training Script
Train the breakthrough model and save for production use
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import train_breakthrough_model


def main():
    """Main training script"""
    print("🚀 TRAINING BREAKTHROUGH REIMBURSEMENT MODEL")
    print("=" * 60)
    
    # Train model
    model = train_breakthrough_model('test_cases.json')
    
    # Save model
    model.save('best_model.pkl', 'training_stats.pkl')
    
    # Display results
    metadata = model.model_metadata
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"📊 Training R²: {metadata['training_r2']:.6f}")
    print(f"🔄 Cross-validation R²: {metadata['cv_r2']:.6f} ± {metadata['cv_std']:.6f}")
    print(f"🔧 Features: {metadata['n_features']}")
    print(f"📁 Model saved to: best_model.pkl")
    print(f"📁 Stats saved to: training_stats.pkl")
    
    if metadata['cv_r2'] >= 0.90:
        print(f"\n✅ EXCELLENT PERFORMANCE! CV R² ≥ 90%")
    elif metadata['cv_r2'] >= 0.85:
        print(f"\n🚀 GOOD PERFORMANCE! CV R² ≥ 85%")
    
    return metadata


if __name__ == "__main__":
    main()