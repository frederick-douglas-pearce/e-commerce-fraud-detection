"""
Human-readable feature descriptions for fraud detection model explanations.

Maps technical feature names to user-friendly descriptions.
"""

FEATURE_DESCRIPTIONS: dict[str, str] = {
    # Original numeric features
    "account_age_days": "Account Age (days)",
    "total_transactions_user": "Total User Transactions",
    "avg_amount_user": "User Average Transaction Amount",
    "amount": "Transaction Amount",
    "shipping_distance_km": "Shipping Distance (km)",
    # Original categorical features
    "channel": "Transaction Channel",
    # Security flags
    "promo_used": "Promo Code Used",
    "avs_match": "Address Verification Match",
    "cvv_result": "CVV Verification Result",
    "three_ds_flag": "3D Secure Enabled",
    # Temporal features (local time)
    "hour_local": "Hour of Day (local)",
    "day_of_week_local": "Day of Week (local)",
    "month_local": "Month",
    "is_weekend_local": "Weekend Transaction",
    "is_late_night_local": "Late Night Transaction (11PM-4AM)",
    "is_business_hours_local": "Business Hours Transaction (9AM-5PM)",
    # Amount features
    "amount_deviation": "Amount Deviation from User Average",
    "amount_vs_avg_ratio": "Amount vs User Average Ratio",
    "is_micro_transaction": "Micro Transaction (under $5)",
    "is_large_transaction": "Large Transaction (top 5%)",
    # User behavior features
    "transaction_velocity": "Transaction Velocity (per day)",
    "is_new_account": "New Account (30 days or less)",
    "is_high_frequency_user": "High Frequency User",
    # Geographic features
    "country_mismatch": "Country Mismatch (card vs transaction)",
    "high_risk_distance": "High Risk Shipping Distance",
    "zero_distance": "Same Billing/Shipping Address",
    # Security score
    "security_score": "Security Score (0-3)",
    # Interaction features (fraud patterns)
    "new_account_with_promo": "New Account Using Promo",
    "late_night_micro_transaction": "Late Night Micro Transaction",
    "high_value_long_distance": "High Value + Long Distance",
}


def get_feature_description(feature_name: str) -> str:
    """
    Get human-readable description for a feature.

    Args:
        feature_name: Technical feature name

    Returns:
        Human-readable description, or the original name if not found
    """
    return FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
