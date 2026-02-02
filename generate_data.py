import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)

def generate_ecommerce_data(n_products=10, n_days=365, n_records_per_day=100):
    """
    Generates synthetic e-commerce pricing data
    """
    
    # Product catalog
    products = []
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Sports', 'Books']
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']
    
    for i in range(n_products):
        products.append({
            'product_id': f'PROD_{i+1:03d}',
            'category': random.choice(categories),
            'brand': random.choice(brands),
            'base_cost': round(random.uniform(10, 200), 2)
        })
    
    # Generate transaction data
    data = []
    start_date = datetime(2023, 1, 1)
    
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        is_weekend = current_date.weekday() >= 5
        is_holiday = current_date.month == 12 or current_date.month == 1  # Simple holiday logic
        
        for _ in range(n_records_per_day):
            product = random.choice(products)
            
            # Base price with markup
            base_price = product['base_cost'] * random.uniform(1.3, 2.0)
            
            # Competitor price (usually similar, sometimes different)
            competitor_price = base_price * random.uniform(0.85, 1.15)
            
            # Time-based demand patterns
            hour = random.randint(0, 23)
            demand_multiplier = 1.0
            
            if is_weekend:
                demand_multiplier *= 1.3
            if is_holiday:
                demand_multiplier *= 1.5
            if 18 <= hour <= 22:  # Evening peak
                demand_multiplier *= 1.4
            elif 9 <= hour <= 17:  # Working hours
                demand_multiplier *= 1.1
            
            # Stock level affects urgency
            stock_level = random.randint(5, 500)
            
            # Customer type
            customer_type = random.choice(['new', 'returning', 'returning', 'returning'])  # 75% returning
            purchase_history = random.randint(0, 20) if customer_type == 'returning' else 0
            
            # Page views (demand indicator)
            base_views = random.randint(50, 500)
            page_views = int(base_views * demand_multiplier)
            
            # Price elasticity effect
            price_ratio = base_price / competitor_price
            elasticity = -1.5  # Typical elasticity
            
            # Calculate demand based on price
            base_demand = random.randint(10, 100) * demand_multiplier
            price_effect = (1 - price_ratio) * elasticity
            adjusted_demand = max(1, int(base_demand * (1 + price_effect)))
            
            # Conversion rate
            conversion_rate = random.uniform(0.02, 0.15) * (1 if stock_level > 50 else 0.7)
            
            # Sales quantity
            potential_sales = int(page_views * conversion_rate)
            sales_quantity = min(potential_sales, adjusted_demand, stock_level)
            
            # Final price (with some random variation)
            final_price = base_price * random.uniform(0.95, 1.05)
            
            # Revenue
            revenue = final_price * sales_quantity
            
            data.append({
                'timestamp': current_date + timedelta(hours=hour, minutes=random.randint(0, 59)),
                'product_id': product['product_id'],
                'category': product['category'],
                'brand': product['brand'],
                'base_cost': product['base_cost'],
                'price': round(final_price, 2),
                'competitor_price': round(competitor_price, 2),
                'stock_level': stock_level,
                'page_views': page_views,
                'cart_adds': int(page_views * random.uniform(0.1, 0.3)),
                'sales_quantity': sales_quantity,
                'revenue': round(revenue, 2),
                'customer_type': customer_type,
                'customer_purchase_history': purchase_history,
                'hour': hour,
                'day_of_week': current_date.weekday(),
                'is_weekend': is_weekend,
                'is_holiday': is_holiday,
                'month': current_date.month
            })
    
    return pd.DataFrame(data)

# Generate data
print("Generating synthetic e-commerce data...")
df = generate_ecommerce_data(n_products=20, n_days=365, n_records_per_day=150)

# Save to CSV
df.to_csv('ecommerce_pricing_data.csv', index=False)
print(f"✓ Generated {len(df)} records")
print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"✓ Products: {df['product_id'].nunique()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData saved to 'ecommerce_pricing_data.csv'")

# Feature Engineering Pipeline
def engineer_features(df):
    """
    Creates all features needed for pricing model
    """
    df = df.copy()
    
    # Time-based features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day_of_month'] = df['timestamp'].dt.day
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    df['quarter'] = df['timestamp'].dt.quarter
    
    # Price features
    df['price_to_cost_ratio'] = df['price'] / df['base_cost']
    df['price_vs_competitor'] = (df['price'] - df['competitor_price']) / df['competitor_price']
    df['margin'] = df['price'] - df['base_cost']
    df['margin_percent'] = (df['margin'] / df['price']) * 100
    
    # Demand indicators
    df['views_to_cart_rate'] = df['cart_adds'] / df['page_views']
    df['cart_to_sale_rate'] = df['sales_quantity'] / (df['cart_adds'] + 1)  # +1 to avoid division by zero
    df['conversion_rate'] = df['sales_quantity'] / (df['page_views'] + 1)
    
    # Inventory pressure
    df['inventory_pressure'] = df['stock_level'].apply(
        lambda x: 'low' if x < 50 else 'medium' if x < 200 else 'high'
    )
    
    # Price elasticity (approximation)
    df['demand_per_dollar'] = df['sales_quantity'] / df['price']
    
    # Revenue per view
    df['revenue_per_view'] = df['revenue'] / (df['page_views'] + 1)
    
    return df

# Apply feature engineering
df_featured = engineer_features(df)
df_featured.to_csv('ecommerce_data_with_features.csv', index=False)
print("✓ Features engineered and saved!")
# Quick data quality check
print("\n=== DATA QUALITY REPORT ===")
print(f"Total records: {len(df_featured)}")
print(f"Missing values:\n{df_featured.isnull().sum()}")
print(f"\nBasic statistics:")
print(df_featured[['price', 'sales_quantity', 'revenue', 'page_views']].describe())

# Save summary stats
summary = df_featured.describe()
summary.to_csv('data_summary_statistics.csv')