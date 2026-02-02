"""
Phase 2: Exploratory Data Analysis (EDA)
Dynamic Pricing Engine for E-commerce

This script performs comprehensive analysis to answer key business questions:
1. Price Elasticity - How does demand respond to price changes?
2. Time Patterns - When do customers buy most?
3. Competitor Impact - How does competitor pricing affect sales?
4. Inventory Pressure - Does stock level affect conversion?
5. Revenue Optimization - What price maximizes revenue?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read the data
print("Loading data...")
df = pd.read_csv('ecommerce_data_with_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"✓ Loaded {len(df)} records")
print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"✓ Products: {df['product_id'].nunique()}\n")

# Create output directory for plots
import os
os.makedirs('eda_visualizations', exist_ok=True)

# ============================================================================
# SECTION 1: PRICE ELASTICITY ANALYSIS
# ============================================================================
print("="*70)
print("SECTION 1: PRICE ELASTICITY ANALYSIS")
print("="*70)

# Calculate price elasticity by product category
def calculate_elasticity(group):
    """Calculate price elasticity of demand"""
    if len(group) < 10:
        return np.nan
    
    # Sort by price
    group = group.sort_values('price')
    
    # Calculate percentage changes
    price_pct_change = group['price'].pct_change()
    demand_pct_change = group['sales_quantity'].pct_change()
    
    # Elasticity = % change in demand / % change in price
    elasticity = demand_pct_change / price_pct_change
    
    return elasticity.replace([np.inf, -np.inf], np.nan).mean()

elasticity_by_category = df.groupby('category').apply(calculate_elasticity)
print("\nPrice Elasticity by Category:")
print(elasticity_by_category.sort_values())
print("\nInterpretation:")
print("- Negative elasticity = Price ↑ → Demand ↓ (normal behavior)")
print("- Elasticity < -1 = Elastic (demand sensitive to price)")
print("- Elasticity > -1 = Inelastic (demand less sensitive)")

# Visualization 1: Price vs Sales Quantity
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Price Elasticity Analysis', fontsize=16, fontweight='bold')

# Plot 1: Overall Price vs Sales
axes[0, 0].scatter(df['price'], df['sales_quantity'], alpha=0.3, s=20)
z = np.polyfit(df['price'], df['sales_quantity'], 2)
p = np.poly1d(z)
x_line = np.linspace(df['price'].min(), df['price'].max(), 100)
axes[0, 0].plot(x_line, p(x_line), "r-", linewidth=2, label='Trend')
axes[0, 0].set_xlabel('Price (₹)', fontsize=12)
axes[0, 0].set_ylabel('Sales Quantity', fontsize=12)
axes[0, 0].set_title('Price vs Sales Quantity (Overall)', fontsize=13)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Price vs Revenue
axes[0, 1].scatter(df['price'], df['revenue'], alpha=0.3, s=20, c='green')
z = np.polyfit(df['price'], df['revenue'], 2)
p = np.poly1d(z)
axes[0, 1].plot(x_line, p(x_line), "r-", linewidth=2, label='Trend')
axes[0, 1].set_xlabel('Price (₹)', fontsize=12)
axes[0, 1].set_ylabel('Revenue (₹)', fontsize=12)
axes[0, 1].set_title('Price vs Revenue (Find the Sweet Spot!)', fontsize=13)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Price Elasticity by Category
elasticity_by_category.plot(kind='barh', ax=axes[1, 0], color='coral')
axes[1, 0].axvline(x=-1, color='red', linestyle='--', label='Elasticity = -1')
axes[1, 0].set_xlabel('Price Elasticity', fontsize=12)
axes[1, 0].set_title('Price Elasticity by Category', fontsize=13)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Plot 4: Conversion Rate vs Price Bins
df['price_bin'] = pd.cut(df['price'], bins=10)
price_conversion = df.groupby('price_bin').agg({
    'conversion_rate': 'mean',
    'sales_quantity': 'count'
}).reset_index()
price_labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in price_conversion['price_bin']]
axes[1, 1].bar(range(len(price_labels)), price_conversion['conversion_rate'], color='teal')
axes[1, 1].set_xticks(range(len(price_labels)))
axes[1, 1].set_xticklabels(price_labels, rotation=45, ha='right')
axes[1, 1].set_xlabel('Price Range (₹)', fontsize=12)
axes[1, 1].set_ylabel('Avg Conversion Rate', fontsize=12)
axes[1, 1].set_title('Conversion Rate by Price Range', fontsize=13)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('eda_visualizations/1_price_elasticity.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_visualizations/1_price_elasticity.png")
plt.close()

# ============================================================================
# SECTION 2: TIME PATTERNS ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("SECTION 2: TIME PATTERNS ANALYSIS")
print("="*70)

# Sales by hour of day
hourly_sales = df.groupby('hour').agg({
    'sales_quantity': 'mean',
    'revenue': 'mean',
    'page_views': 'mean',
    'conversion_rate': 'mean'
}).reset_index()

print("\nPeak Hours for Sales:")
print(hourly_sales.nlargest(5, 'sales_quantity')[['hour', 'sales_quantity', 'conversion_rate']])

# Sales by day of week
daily_sales = df.groupby('day_of_week').agg({
    'sales_quantity': 'mean',
    'revenue': 'mean',
    'conversion_rate': 'mean'
}).reset_index()
daily_sales['day_name'] = daily_sales['day_of_week'].map({
    0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
})

print("\nSales by Day of Week:")
print(daily_sales[['day_name', 'sales_quantity', 'conversion_rate']])

# Weekend vs Weekday comparison
weekend_comparison = df.groupby('is_weekend').agg({
    'sales_quantity': 'mean',
    'revenue': 'mean',
    'conversion_rate': 'mean',
    'page_views': 'mean'
}).reset_index()
weekend_comparison['period'] = weekend_comparison['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})

print("\nWeekend vs Weekday Performance:")
print(weekend_comparison[['period', 'sales_quantity', 'revenue', 'conversion_rate']])

# Visualization 2: Time Patterns
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Time Pattern Analysis - When Do Customers Buy?', fontsize=16, fontweight='bold')

# Plot 1: Hourly sales pattern
axes[0, 0].plot(hourly_sales['hour'], hourly_sales['sales_quantity'], marker='o', linewidth=2, markersize=8)
axes[0, 0].fill_between(hourly_sales['hour'], hourly_sales['sales_quantity'], alpha=0.3)
axes[0, 0].set_xlabel('Hour of Day', fontsize=12)
axes[0, 0].set_ylabel('Avg Sales Quantity', fontsize=12)
axes[0, 0].set_title('Sales Pattern by Hour', fontsize=13)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(range(0, 24, 2))

# Plot 2: Daily pattern
axes[0, 1].bar(daily_sales['day_name'], daily_sales['sales_quantity'], color='skyblue', edgecolor='navy')
axes[0, 1].set_xlabel('Day of Week', fontsize=12)
axes[0, 1].set_ylabel('Avg Sales Quantity', fontsize=12)
axes[0, 1].set_title('Sales Pattern by Day of Week', fontsize=13)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Weekend vs Weekday comparison
x_pos = [0, 1]
weekend_metrics = weekend_comparison.set_index('period')
width = 0.25
axes[1, 0].bar([p - width for p in x_pos], 
               [weekend_metrics.loc['Weekday', 'sales_quantity'], weekend_metrics.loc['Weekend', 'sales_quantity']], 
               width, label='Sales', color='coral')
axes[1, 0].bar(x_pos, 
               [weekend_metrics.loc['Weekday', 'page_views']/10, weekend_metrics.loc['Weekend', 'page_views']/10], 
               width, label='Views (÷10)', color='lightblue')
axes[1, 0].bar([p + width for p in x_pos], 
               [weekend_metrics.loc['Weekday', 'conversion_rate']*100, weekend_metrics.loc['Weekend', 'conversion_rate']*100], 
               width, label='Conv% (×100)', color='lightgreen')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(['Weekday', 'Weekend'])
axes[1, 0].set_ylabel('Scaled Metrics', fontsize=12)
axes[1, 0].set_title('Weekend vs Weekday Performance', fontsize=13)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Conversion rate by hour
axes[1, 1].plot(hourly_sales['hour'], hourly_sales['conversion_rate']*100, 
                marker='s', linewidth=2, markersize=8, color='green')
axes[1, 1].fill_between(hourly_sales['hour'], hourly_sales['conversion_rate']*100, alpha=0.3, color='green')
axes[1, 1].set_xlabel('Hour of Day', fontsize=12)
axes[1, 1].set_ylabel('Conversion Rate (%)', fontsize=12)
axes[1, 1].set_title('Conversion Rate by Hour', fontsize=13)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig('eda_visualizations/2_time_patterns.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_visualizations/2_time_patterns.png")
plt.close()

# ============================================================================
# SECTION 3: COMPETITOR PRICING IMPACT
# ============================================================================
print("\n" + "="*70)
print("SECTION 3: COMPETITOR PRICING IMPACT")
print("="*70)

# Price positioning analysis
df['price_position'] = pd.cut(df['price_vs_competitor'], 
                              bins=[-np.inf, -0.1, 0.1, np.inf],
                              labels=['Cheaper', 'Similar', 'Expensive'])

position_analysis = df.groupby('price_position').agg({
    'sales_quantity': ['mean', 'count'],
    'revenue': 'mean',
    'conversion_rate': 'mean'
}).reset_index()

print("\nPerformance by Price Position vs Competitor:")
print(position_analysis)

# Detailed competitor impact
df['competitor_diff'] = df['price'] - df['competitor_price']
competitor_bins = pd.cut(df['competitor_diff'], bins=10)
competitor_impact = df.groupby(competitor_bins).agg({
    'sales_quantity': 'mean',
    'revenue': 'mean',
    'conversion_rate': 'mean'
}).reset_index()

# Visualization 3: Competitor Impact
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Competitor Pricing Impact Analysis', fontsize=16, fontweight='bold')

# Plot 1: Your Price vs Competitor Price
axes[0, 0].scatter(df['competitor_price'], df['price'], alpha=0.3, s=20)
axes[0, 0].plot([df['competitor_price'].min(), df['competitor_price'].max()], 
                [df['competitor_price'].min(), df['competitor_price'].max()], 
                'r--', linewidth=2, label='Equal Pricing')
axes[0, 0].set_xlabel('Competitor Price (₹)', fontsize=12)
axes[0, 0].set_ylabel('Your Price (₹)', fontsize=12)
axes[0, 0].set_title('Your Price vs Competitor Price', fontsize=13)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Sales by price position
position_sales = df.groupby('price_position')['sales_quantity'].mean()
colors_map = {'Cheaper': 'green', 'Similar': 'orange', 'Expensive': 'red'}
axes[0, 1].bar(position_sales.index, position_sales.values, 
               color=[colors_map[x] for x in position_sales.index])
axes[0, 1].set_xlabel('Price Position', fontsize=12)
axes[0, 1].set_ylabel('Avg Sales Quantity', fontsize=12)
axes[0, 1].set_title('Sales by Price Position vs Competitor', fontsize=13)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Revenue by price position
position_revenue = df.groupby('price_position')['revenue'].mean()
axes[1, 0].bar(position_revenue.index, position_revenue.values, 
               color=[colors_map[x] for x in position_revenue.index])
axes[1, 0].set_xlabel('Price Position', fontsize=12)
axes[1, 0].set_ylabel('Avg Revenue (₹)', fontsize=12)
axes[1, 0].set_title('Revenue by Price Position', fontsize=13)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Conversion rate by price position
position_conversion = df.groupby('price_position')['conversion_rate'].mean() * 100
axes[1, 1].bar(position_conversion.index, position_conversion.values, 
               color=[colors_map[x] for x in position_conversion.index])
axes[1, 1].set_xlabel('Price Position', fontsize=12)
axes[1, 1].set_ylabel('Conversion Rate (%)', fontsize=12)
axes[1, 1].set_title('Conversion Rate by Price Position', fontsize=13)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('eda_visualizations/3_competitor_impact.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_visualizations/3_competitor_impact.png")
plt.close()

# ============================================================================
# SECTION 4: INVENTORY PRESSURE ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("SECTION 4: INVENTORY PRESSURE ANALYSIS")
print("="*70)

# Create inventory categories
df['stock_category'] = pd.cut(df['stock_level'], 
                              bins=[0, 50, 100, 200, np.inf],
                              labels=['Critical (<50)', 'Low (50-100)', 'Medium (100-200)', 'High (>200)'])

inventory_analysis = df.groupby('stock_category').agg({
    'sales_quantity': 'mean',
    'conversion_rate': 'mean',
    'price': 'mean',
    'revenue': 'mean'
}).reset_index()

print("\nPerformance by Inventory Level:")
print(inventory_analysis)

# Correlation between stock and sales
stock_sales_corr = df['stock_level'].corr(df['sales_quantity'])
print(f"\nCorrelation between Stock Level and Sales: {stock_sales_corr:.3f}")

# Visualization 4: Inventory Impact
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Inventory Pressure Analysis', fontsize=16, fontweight='bold')

# Plot 1: Stock level vs Sales
axes[0, 0].scatter(df['stock_level'], df['sales_quantity'], alpha=0.3, s=20)
z = np.polyfit(df['stock_level'], df['sales_quantity'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['stock_level'].min(), df['stock_level'].max(), 100)
axes[0, 0].plot(x_line, p(x_line), "r-", linewidth=2, label=f'Trend (corr={stock_sales_corr:.2f})')
axes[0, 0].set_xlabel('Stock Level', fontsize=12)
axes[0, 0].set_ylabel('Sales Quantity', fontsize=12)
axes[0, 0].set_title('Stock Level vs Sales', fontsize=13)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Conversion by stock category
axes[0, 1].bar(range(len(inventory_analysis)), 
               inventory_analysis['conversion_rate']*100, 
               color='purple', alpha=0.7)
axes[0, 1].set_xticks(range(len(inventory_analysis)))
axes[0, 1].set_xticklabels(inventory_analysis['stock_category'], rotation=45, ha='right')
axes[0, 1].set_ylabel('Conversion Rate (%)', fontsize=12)
axes[0, 1].set_title('Conversion Rate by Inventory Level', fontsize=13)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Average price by stock level
axes[1, 0].bar(range(len(inventory_analysis)), 
               inventory_analysis['price'], 
               color='orange', alpha=0.7)
axes[1, 0].set_xticks(range(len(inventory_analysis)))
axes[1, 0].set_xticklabels(inventory_analysis['stock_category'], rotation=45, ha='right')
axes[1, 0].set_ylabel('Average Price (₹)', fontsize=12)
axes[1, 0].set_title('Pricing Strategy by Inventory Level', fontsize=13)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Revenue by stock category
axes[1, 1].bar(range(len(inventory_analysis)), 
               inventory_analysis['revenue'], 
               color='teal', alpha=0.7)
axes[1, 1].set_xticks(range(len(inventory_analysis)))
axes[1, 1].set_xticklabels(inventory_analysis['stock_category'], rotation=45, ha='right')
axes[1, 1].set_ylabel('Average Revenue (₹)', fontsize=12)
axes[1, 1].set_title('Revenue by Inventory Level', fontsize=13)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('eda_visualizations/4_inventory_pressure.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_visualizations/4_inventory_pressure.png")
plt.close()

# ============================================================================
# SECTION 5: REVENUE OPTIMIZATION & CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("SECTION 5: REVENUE OPTIMIZATION & FEATURE CORRELATIONS")
print("="*70)

# Find optimal price point for maximum revenue by category
optimal_prices = []
for category in df['category'].unique():
    cat_data = df[df['category'] == category].copy()
    # Group by price bins and calculate average revenue
    price_bins = pd.cut(cat_data['price'], bins=20)
    revenue_by_price = cat_data.groupby(price_bins)['revenue'].mean()
    optimal_bin = revenue_by_price.idxmax()
    optimal_price = (optimal_bin.left + optimal_bin.right) / 2
    max_revenue = revenue_by_price.max()
    optimal_prices.append({
        'category': category,
        'optimal_price': optimal_price,
        'max_revenue': max_revenue,
        'avg_price': cat_data['price'].mean()
    })

optimal_df = pd.DataFrame(optimal_prices)
print("\nOptimal Pricing by Category:")
print(optimal_df)

# Correlation matrix for key features
correlation_features = [
    'price', 'competitor_price', 'stock_level', 'page_views',
    'sales_quantity', 'revenue', 'conversion_rate', 
    'price_vs_competitor', 'margin_percent'
]
correlation_matrix = df[correlation_features].corr()

print("\nTop Positive Correlations with Revenue:")
revenue_corr = correlation_matrix['revenue'].sort_values(ascending=False)
print(revenue_corr[1:6])  # Exclude self-correlation

print("\nTop Negative Correlations with Revenue:")
print(revenue_corr[-5:])

# Visualization 5: Revenue Optimization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Revenue Optimization & Correlation Analysis', fontsize=16, fontweight='bold')

# Plot 1: Optimal price by category
x_pos = range(len(optimal_df))
axes[0, 0].bar([p - 0.2 for p in x_pos], optimal_df['avg_price'], 
               width=0.4, label='Current Avg Price', color='lightblue')
axes[0, 0].bar([p + 0.2 for p in x_pos], optimal_df['optimal_price'], 
               width=0.4, label='Optimal Price', color='green')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(optimal_df['category'], rotation=45, ha='right')
axes[0, 0].set_ylabel('Price (₹)', fontsize=12)
axes[0, 0].set_title('Current vs Optimal Price by Category', fontsize=13)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: Correlation heatmap
im = axes[0, 1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
axes[0, 1].set_xticks(range(len(correlation_features)))
axes[0, 1].set_yticks(range(len(correlation_features)))
axes[0, 1].set_xticklabels([f.replace('_', '\n') for f in correlation_features], 
                           rotation=45, ha='right', fontsize=9)
axes[0, 1].set_yticklabels([f.replace('_', ' ') for f in correlation_features], fontsize=9)
axes[0, 1].set_title('Feature Correlation Heatmap', fontsize=13)
plt.colorbar(im, ax=axes[0, 1])

# Plot 3: Revenue vs margin percentage
axes[1, 0].scatter(df['margin_percent'], df['revenue'], alpha=0.3, s=20)
z = np.polyfit(df['margin_percent'], df['revenue'], 2)
p = np.poly1d(z)
x_line = np.linspace(df['margin_percent'].min(), df['margin_percent'].max(), 100)
axes[1, 0].plot(x_line, p(x_line), "r-", linewidth=2, label='Trend')
axes[1, 0].set_xlabel('Margin % (Profit/Price)', fontsize=12)
axes[1, 0].set_ylabel('Revenue (₹)', fontsize=12)
axes[1, 0].set_title('Revenue vs Profit Margin', fontsize=13)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Feature importance for revenue
revenue_corr_abs = correlation_matrix['revenue'].abs().sort_values(ascending=False)[1:9]
axes[1, 1].barh(range(len(revenue_corr_abs)), revenue_corr_abs.values, color='steelblue')
axes[1, 1].set_yticks(range(len(revenue_corr_abs)))
axes[1, 1].set_yticklabels([f.replace('_', ' ').title() for f in revenue_corr_abs.index])
axes[1, 1].set_xlabel('Absolute Correlation with Revenue', fontsize=12)
axes[1, 1].set_title('Top Features Correlated with Revenue', fontsize=13)
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('eda_visualizations/5_revenue_optimization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_visualizations/5_revenue_optimization.png")
plt.close()

# ============================================================================
# SECTION 6: CUSTOMER SEGMENTATION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("SECTION 6: CUSTOMER SEGMENTATION ANALYSIS")
print("="*70)

customer_analysis = df.groupby('customer_type').agg({
    'sales_quantity': 'mean',
    'revenue': 'mean',
    'conversion_rate': 'mean',
    'price': 'mean',
    'product_id': 'count'
}).reset_index()
customer_analysis.columns = ['customer_type', 'avg_sales', 'avg_revenue', 
                             'conversion_rate', 'avg_price', 'total_transactions']

print("\nCustomer Type Performance:")
print(customer_analysis)

# Visualization 6: Customer Segmentation
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold')

# Plot 1: Transaction distribution
axes[0, 0].pie(customer_analysis['total_transactions'], 
               labels=customer_analysis['customer_type'],
               autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue'])
axes[0, 0].set_title('Transaction Distribution by Customer Type', fontsize=13)

# Plot 2: Revenue per customer type
axes[0, 1].bar(customer_analysis['customer_type'], customer_analysis['avg_revenue'], 
               color=['coral', 'skyblue'])
axes[0, 1].set_ylabel('Average Revenue (₹)', fontsize=12)
axes[0, 1].set_title('Revenue by Customer Type', fontsize=13)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Conversion rate comparison
axes[1, 0].bar(customer_analysis['customer_type'], 
               customer_analysis['conversion_rate']*100, 
               color=['coral', 'skyblue'])
axes[1, 0].set_ylabel('Conversion Rate (%)', fontsize=12)
axes[1, 0].set_title('Conversion Rate by Customer Type', fontsize=13)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Customer lifetime comparison (for returning customers)
returning_customers = df[df['customer_type'] == 'returning']
purchase_history_impact = returning_customers.groupby(
    pd.cut(returning_customers['customer_purchase_history'], bins=5)
).agg({
    'revenue': 'mean',
    'sales_quantity': 'mean'
}).reset_index()

axes[1, 1].plot(range(len(purchase_history_impact)), 
                purchase_history_impact['revenue'], 
                marker='o', linewidth=2, markersize=8, color='purple')
axes[1, 1].set_xlabel('Purchase History Level', fontsize=12)
axes[1, 1].set_ylabel('Average Revenue (₹)', fontsize=12)
axes[1, 1].set_title('Revenue vs Customer Purchase History', fontsize=13)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_visualizations/6_customer_segmentation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_visualizations/6_customer_segmentation.png")
plt.close()

# ============================================================================
# FINAL SUMMARY & KEY INSIGHTS
# ============================================================================
print("\n" + "="*70)
print("KEY INSIGHTS SUMMARY")
print("="*70)

insights = {
    '1. Price Elasticity': f"Average elasticity: {df.groupby('category').apply(calculate_elasticity).mean():.2f}",
    '2. Best Time to Sell': f"Peak hour: {hourly_sales.loc[hourly_sales['sales_quantity'].idxmax(), 'hour']}:00",
    '3. Weekend Effect': f"Weekend sales {weekend_comparison.loc[1, 'sales_quantity']/weekend_comparison.loc[0, 'sales_quantity']*100-100:.1f}% higher than weekday",
    '4. Price Position': f"Best position: {position_analysis.loc[position_analysis[('sales_quantity', 'mean')].idxmax(), 'price_position']}",
    '5. Inventory Impact': f"Stock-sales correlation: {stock_sales_corr:.3f}",
    '6. Customer Value': f"Returning customers generate {customer_analysis.loc[customer_analysis['customer_type']=='returning', 'avg_revenue'].values[0]/customer_analysis.loc[customer_analysis['customer_type']=='new', 'avg_revenue'].values[0]:.2f}x revenue",
    '7. Revenue Driver': f"Top correlation: {revenue_corr.index[1]} ({revenue_corr.iloc[1]:.3f})"
}

print("\n📊 Key Business Insights:")
for key, value in insights.items():
    print(f"   {key}: {value}")

# Save insights to file
with open('eda_visualizations/key_insights.txt', 'w') as f:
    f.write("PHASE 2: KEY INSIGHTS FROM EXPLORATORY DATA ANALYSIS\n")
    f.write("="*70 + "\n\n")
    for key, value in insights.items():
        f.write(f"{key}: {value}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("RECOMMENDATIONS FOR PRICING STRATEGY:\n")
    f.write("="*70 + "\n")
    f.write("1. Implement time-based pricing (increase during peak hours)\n")
    f.write("2. Adjust prices based on competitor positioning\n")
    f.write("3. Use urgency pricing when inventory is low\n")
    f.write("4. Offer targeted discounts for new customers\n")
    f.write("5. Focus on high-margin products during weekends\n")

print("\n✓ All visualizations saved in 'eda_visualizations/' folder")
print("✓ Key insights saved to 'eda_visualizations/key_insights.txt'")
print("\n" + "="*70)
print("PHASE 2 COMPLETE! Ready to move to Phase 3: Model Building")
print("="*70)