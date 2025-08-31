# Enhanced ML Submission - Target-Specific Modeling

## Brief Description

This submission implements target-specific machine learning models for all 424 financial targets using ensemble methods and advanced feature engineering.

## Key Features:
- **Target-specific modeling** using target_pairs.csv analysis
- **Ensemble approach**: Ridge + Random Forest + XGBoost
- **Feature engineering**: Technical indicators, cross-asset spreads, FX relationships
- **Robust data handling** with proper NaN and outlier management

## Technical Approach:
- Individual models trained for each target based on underlying asset relationships
- Feature selection using competition data insights (LME, JPX, US_Stock, FX)
- Ensemble predictions for stability and reduced overfitting
- Conservative prediction strategy with fallback mechanisms

## Expected Improvements:
- More realistic predictions based on actual market relationships
- Better generalization through proper feature engineering
- Robust handling of missing data and edge cases

**Model Type**: Ensemble ML with Feature Engineering  
**Targets**: All 424 financial instrument predictions  
**Format**: submission.parquet 