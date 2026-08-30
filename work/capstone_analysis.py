"""
FlyRank Capstone: Ranking Signal Analysis
This script reads the FlyRank data, finds what makes pages get more clicks,
and saves 10 charts + results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# === SETUP FOLDERS ===
BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"
for d in [OUTPUT_DIR, CHARTS_DIR]:
    d.mkdir(exist_ok=True)

# === 1. LOAD DATA ===
df = pd.read_csv(BASE_DIR / "data/raw/content_refresh_anonymized.csv")
print(f"Loaded: {len(df):,} rows")

# === 2. FILTER (Data Contract) ===
analysis_df = df[
    (df['impressions_90d'] >= 100) & 
    (df['avg_position'] > 0) &
    (df['sessions_90d'] > 0)
].copy()
print(f"After filtering: {len(analysis_df):,} rows")

# === 3. CREATE FEATURES ===
for col in ['impressions_90d', 'clicks_90d', 'sessions_90d', 'ai_sessions_90d']:
    analysis_df[f'log_{col}'] = np.log1p(analysis_df[col])

analysis_df['has_clicks'] = (analysis_df['clicks_90d'] > 0).astype(int)
analysis_df['has_ai_sessions'] = (analysis_df['ai_sessions_90d'] > 0).astype(int)
analysis_df['measurable_opportunity'] = ((analysis_df['impressions_90d'] >= 100) & (analysis_df['sessions_90d'] > 0)).astype(int)

for col in ['competition_level', 'main_intent', 'word_count_tier', 'char_count_tier', 'freshness_tier']:
    analysis_df[col] = analysis_df[col].fillna('unknown')
for col in ['search_volume', 'competition', 'cpc', 'word_count', 'char_count']:
    analysis_df[col] = analysis_df[col].fillna(0)

# === 4. CREATE LABEL ===
def label_ctr_tier(group):
    q75 = group['ctr'].quantile(0.75)
    q25 = group['ctr'].quantile(0.25)
    group['ctr_label'] = np.where(group['ctr'] >= q75, 'high',
                          np.where(group['ctr'] <= q25, 'low', 'mid'))
    return group

analysis_df = analysis_df.groupby('position_tier', group_keys=False).apply(label_ctr_tier)
analysis_df['ctr_label_binary'] = np.where(analysis_df['ctr_label'] == 'high', 1,
                                   np.where(analysis_df['ctr_label'] == 'low', 0, np.nan))
model_df = analysis_df[analysis_df['ctr_label_binary'].notna()].copy()
model_df['ctr_label_binary'] = model_df['ctr_label_binary'].astype(int)
print(f"Pages used for modeling: {len(model_df):,}")

# === 5. CHART 1: CTR by Position Tier ===
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
tiers = ['top_3', 'page_1', 'striking', 'page_3_5', 'deep']
for i, tier in enumerate(tiers):
    ax = axes[i]
    data = analysis_df[analysis_df['position_tier'] == tier]['ctr']
    ax.hist(data, bins=50, color='#2563eb', edgecolor='white', alpha=0.8)
    ax.axvline(data.quantile(0.75), color='#dc2626', linestyle='--', linewidth=2, label=f'75th: {data.quantile(0.75):.2f}%')
    ax.axvline(data.quantile(0.25), color='#16a34a', linestyle='--', linewidth=2, label=f'25th: {data.quantile(0.25):.2f}%')
    ax.set_title(f'{tier.replace("_", " ").title()} (n={len(data):,})', fontsize=12, fontweight='bold')
    ax.set_xlabel('CTR (%)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(data.quantile(0.99) * 1.5, 20))
axes[5].axis('off')
fig.suptitle('CTR Distribution by Position Tier', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(CHARTS_DIR / '01_ctr_distribution_by_tier.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1 saved")

# === 6. CHART 2: Signal Profiles ===
signal_cols = ['word_count', 'content_age_days', 'days_since_last_update', 
               'engagement_rate', 'scroll_rate', 'search_volume', 'competition']
profile_data = []
for tier in model_df['position_tier'].unique():
    tier_df = model_df[model_df['position_tier'] == tier].copy()
    for col in signal_cols:
        tier_df[f'{col}_z'] = (tier_df[col] - tier_df[col].mean()) / tier_df[col].std()
    high = tier_df[tier_df['ctr_label_binary'] == 1]
    low = tier_df[tier_df['ctr_label_binary'] == 0]
    for col in signal_cols:
        profile_data.append({
            'position_tier': tier, 'signal': col,
            'high_ctr_mean': high[f'{col}_z'].mean(),
            'low_ctr_mean': low[f'{col}_z'].mean(),
            'diff': high[f'{col}_z'].mean() - low[f'{col}_z'].mean()
        })
profile_df = pd.DataFrame(profile_data)
tier_weights = model_df['position_tier'].value_counts().to_dict()
profile_agg = profile_df.groupby('signal').apply(
    lambda x: pd.Series({
        'high_ctr': np.average(x['high_ctr_mean'], weights=x['position_tier'].map(tier_weights)),
        'low_ctr': np.average(x['low_ctr_mean'], weights=x['position_tier'].map(tier_weights)),
        'diff': np.average(x['diff'], weights=x['position_tier'].map(tier_weights))
    })
).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(profile_agg))
width = 0.35
ax.bar(x - width/2, profile_agg['high_ctr'], width, label='High CTR', color='#16a34a', edgecolor='white')
ax.bar(x + width/2, profile_agg['low_ctr'], width, label='Low CTR', color='#dc2626', edgecolor='white')
ax.set_ylabel('Standardized Signal Strength')
ax.set_title('Signal Profiles: High-CTR vs Low-CTR Pages', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([s.replace('_', ' ').title() for s in profile_agg['signal']], rotation=30, ha='right')
ax.legend()
ax.axhline(0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)
for i, row in profile_agg.iterrows():
    ax.annotate(f"Δ={row['diff']:+.2f}", xy=(i, max(row['high_ctr'], row['low_ctr']) + 0.05),
                ha='center', fontsize=9, fontweight='bold', color='#4b5563')
plt.tight_layout()
plt.savefig(CHARTS_DIR / '02_signal_profiles_high_vs_low.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2 saved")

# === 7. CHART 3: Correlation Heatmap ===
corr_cols = ['ctr', 'avg_position', 'word_count', 'content_age_days', 'days_since_last_update',
             'engagement_rate', 'scroll_rate', 'search_volume', 'competition', 'cpc',
             'impressions_90d', 'clicks_90d', 'sessions_90d']
corr_matrix = model_df[corr_cols].corr(method='spearman')
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='Spearman Correlation')
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels([c.replace('_', ' ').title() for c in corr_cols], rotation=45, ha='right')
ax.set_yticklabels([c.replace('_', ' ').title() for c in corr_cols])
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center',
               fontsize=8, color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
ax.set_title('Spearman Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(CHARTS_DIR / '03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3 saved")

# === 8. CHART 4: CTR by Content Type & Intent ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ct_data = analysis_df.groupby('content_type')['ctr'].agg(['median', 'mean', 'count']).reset_index().sort_values('median')
axes[0].barh(ct_data['content_type'], ct_data['median'], color='#3b82f6', edgecolor='white')
axes[0].set_xlabel('Median CTR (%)')
axes[0].set_title('Median CTR by Content Type', fontweight='bold')
for i, row in ct_data.iterrows():
    axes[0].text(row['median'] + 0.05, i, f"n={row['count']:,}", va='center', fontsize=9)
mi_data = analysis_df[analysis_df['main_intent'].notna()].groupby('main_intent')['ctr'].agg(['median', 'mean', 'count']).reset_index().sort_values('median')
axes[1].barh(mi_data['main_intent'], mi_data['median'], color='#8b5cf6', edgecolor='white')
axes[1].set_xlabel('Median CTR (%)')
axes[1].set_title('Median CTR by Search Intent', fontweight='bold')
for i, row in mi_data.iterrows():
    axes[1].text(row['median'] + 0.05, i, f"n={row['count']:,}", va='center', fontsize=9)
plt.suptitle('CTR by Content Classification', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(CHARTS_DIR / '04_ctr_by_type_and_intent.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 4 saved")

# === 9. CHART 5: CTR vs Word Count ===
analysis_df['word_count_bin'] = pd.cut(analysis_df['word_count'], bins=10)
wc_data = analysis_df.groupby('word_count_bin')['ctr'].agg(['median', 'mean', 'count', 'std']).reset_index()
wc_data['bin_center'] = wc_data['word_count_bin'].apply(lambda x: x.mid)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(wc_data['bin_center'], wc_data['median'], s=wc_data['count']*2, 
           c='#059669', alpha=0.7, edgecolors='white', linewidth=1)
ax.plot(wc_data['bin_center'], wc_data['median'], color='#059669', linewidth=2, alpha=0.5)
ax.set_xlabel('Word Count (bin center)')
ax.set_ylabel('Median CTR (%)')
ax.set_title('CTR vs Word Count (Bubble size = row count)', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CHARTS_DIR / '05_ctr_vs_wordcount.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 5 saved")

# === 10. BASELINE SCORE ===
def baseline_score(row):
    score = 0
    if row['engagement_rate'] > 50: score += 0.25
    if row['scroll_rate'] > 60: score += 0.15
    if row['days_since_last_update'] <= 90: score += 0.20
    if row['word_count'] >= 1500: score += 0.15
    if row['position_tier'] in ['top_3', 'page_1']: score += 0.15
    if row['search_volume'] > 1000: score += 0.10
    return score

model_df['baseline_score'] = model_df.apply(baseline_score, axis=1)

# === 11. PREPARE MODELING ===
NUMERIC_FEATURES = [
    'search_volume', 'competition', 'cpc', 'word_count', 'char_count',
    'content_age_days', 'days_since_last_update',
    'log_impressions_90d', 'log_clicks_90d', 'log_sessions_90d', 'log_ai_sessions_90d',
    'days_with_impressions', 'days_with_sessions',
    'engagement_rate', 'scroll_rate', 'ai_traffic_pct',
    'has_clicks', 'has_ai_sessions', 'measurable_opportunity'
]
CATEGORICAL_FEATURES = [
    'competition_level', 'content_type', 'main_intent',
    'age_tier', 'freshness_tier', 'word_count_tier', 'impression_tier', 'position_tier'
]
# Fill any remaining NaN in numeric features with 0
model_df[NUMERIC_FEATURES] = model_df[NUMERIC_FEATURES].fillna(0)

y = model_df['ctr_label_binary'].values
groups = model_df['client_id'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(model_df, y, groups))
X_train_raw = model_df.iloc[train_idx]
X_test_raw = model_df.iloc[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUMERIC_FEATURES),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
])
preprocessor.fit(X_train_raw[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
X_train = preprocessor.transform(X_train_raw[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
X_test = preprocessor.transform(X_test_raw[NUMERIC_FEATURES + CATEGORICAL_FEATURES])

print(f"Training on: {len(X_train_raw):,} rows from {X_train_raw['client_id'].nunique()} clients")
print(f"Testing on:  {len(X_test_raw):,} rows from {X_test_raw['client_id'].nunique()} clients")

# === 12. TRAIN MODELS ===
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20, 
                                            class_weight='balanced', random_state=42, n_jobs=-1)
}

results = {}

baseline_auc = roc_auc_score(y_test, X_test_raw['baseline_score'].values)
baseline_ap = average_precision_score(y_test, X_test_raw['baseline_score'].values)
baseline_p50 = X_test_raw.nlargest(50, 'baseline_score')['ctr_label_binary'].mean()
results['Baseline Rules'] = {'auc': baseline_auc, 'ap': baseline_ap, 'p50': baseline_p50}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    test_df = X_test_raw.copy()
    test_df['proba'] = y_proba
    p50 = test_df.nlargest(50, 'proba')['ctr_label_binary'].mean()
    results[name] = {'auc': auc, 'ap': ap, 'p50': p50, 'model': clf, 'proba': y_proba}
    print(f"{name:20s}: AUC={auc:.3f}, AP={ap:.3f}, P@50={p50:.3f}")

print(f"\nBaseline Rules: AUC={baseline_auc:.3f}, AP={baseline_ap:.3f}, P@50={baseline_p50:.3f}")

# === 13. CHART 6: Model Comparison ===
fig, ax = plt.subplots(figsize=(10, 6))
methods = list(results.keys())
aucs = [results[m]['auc'] for m in methods]
aps = [results[m]['ap'] for m in methods]
p50s = [results[m]['p50'] for m in methods]
x = np.arange(len(methods))
width = 0.25
bars1 = ax.bar(x - width, aucs, width, label='ROC AUC', color='#3b82f6', edgecolor='white')
bars2 = ax.bar(x, aps, width, label='Avg Precision', color='#8b5cf6', edgecolor='white')
bars3 = ax.bar(x + width, p50s, width, label='Precision@50', color='#10b981', edgecolor='white')
ax.set_ylabel('Score')
ax.set_title('Model Performance: Client-Holdout Validation', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha='right')
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(CHARTS_DIR / '06_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 6 saved")

# === 14. CHART 7: Feature Importance ===
rf = results['Random Forest']['model']
feature_names = (NUMERIC_FEATURES + list(preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)))
fi_df = pd.DataFrame({'feature': feature_names, 'importance': rf.feature_importances_})
fi_df = fi_df.sort_values('importance', ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(fi_df['feature'], fi_df['importance'], color='#f59e0b', edgecolor='white')
ax.set_xlabel('Importance')
ax.set_title('Top 20 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(CHARTS_DIR / '07_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 7 saved")

# === 15. CHART 8: ROC Curves ===
fig, ax = plt.subplots(figsize=(10, 8))
colors = {'Baseline Rules': '#9ca3af', 'Logistic Regression': '#3b82f6', 
          'Decision Tree': '#f59e0b', 'Random Forest': '#10b981'}
fpr, tpr, _ = roc_curve(y_test, X_test_raw['baseline_score'].values)
ax.plot(fpr, tpr, color=colors['Baseline Rules'], linewidth=2, label=f"Baseline (AUC={results['Baseline Rules']['auc']:.3f})")
for name in ['Logistic Regression', 'Decision Tree', 'Random Forest']:
    fpr, tpr, _ = roc_curve(y_test, results[name]['proba'])
    ax.plot(fpr, tpr, color=colors[name], linewidth=2, label=f"{name} (AUC={results[name]['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves: Client-Holdout Test Set', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CHARTS_DIR / '08_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 8 saved")

# === 16. CHART 9: Precision@K ===
fig, ax = plt.subplots(figsize=(10, 6))
k_values = range(10, 201, 10)
for name in ['Baseline Rules', 'Logistic Regression', 'Decision Tree', 'Random Forest']:
    precisions = []
    test_df = X_test_raw.copy()
    if name == 'Baseline Rules':
        test_df['score'] = test_df['baseline_score']
    else:
        test_df['score'] = results[name]['proba']
    for k in k_values:
        topk = test_df.nlargest(k, 'score')
        precisions.append(topk['ctr_label_binary'].mean())
    ax.plot(k_values, precisions, color=colors[name], linewidth=2, marker='o', markersize=4, label=name)
ax.set_xlabel('K (Top-K Reviewed)')
ax.set_ylabel('Precision@K')
ax.set_title('Precision@K: How Many Top Predictions Are Correct?', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(CHARTS_DIR / '09_precision_at_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 9 saved")

# === 17. CHART 10: Action Queue Top 20 ===
test_df = X_test_raw.copy()
test_df['final_score'] = results['Random Forest']['proba']
conditions = [
    (test_df['position_tier'].isin(['top_3', 'page_1'])) & (test_df['ctr'] < 1.0),
    (test_df['engagement_rate'] > 50) & (test_df['ctr'] < 1.0),
    (test_df['word_count'] < 1200) & (test_df['impressions_90d'] >= 500),
    (test_df['days_since_last_update'] > 180) & (test_df['impressions_90d'] >= 500),
    (test_df['scroll_rate'] < 30) & (test_df['sessions_90d'] >= 30)
]
reasons = ['page_one_underperformer', 'high_engagement_low_ctr', 'thin_visible_page',
           'stale_visible_page', 'low_scroll_engagement']
test_df['reason_code'] = np.select(conditions, reasons, default='general_review')
top20 = test_df.nlargest(20, 'final_score')[['content_id', 'client_id', 'position_tier', 
                                              'ctr', 'avg_position', 'engagement_rate', 
                                              'word_count', 'final_score', 'reason_code']]

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')
table_data = top20.copy()
table_data['final_score'] = table_data['final_score'].round(3)
table_data['ctr'] = table_data['ctr'].round(2)
table_data['engagement_rate'] = table_data['engagement_rate'].round(1)
table_data.columns = ['Content ID', 'Client', 'Position', 'CTR%', 'Avg Pos', 'Engage%', 'Words', 'Score', 'Reason']
table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                cellLoc='center', loc='center', colWidths=[0.15, 0.12, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)
for i in range(len(table_data.columns)):
    table[(0, i)].set_facecolor('#1e40af')
    table[(0, i)].set_text_props(weight='bold', color='white')
for i in range(1, len(table_data) + 1):
    for j in range(len(table_data.columns)):
        table[(i, j)].set_facecolor('#f8fafc' if i % 2 == 0 else '#ffffff')
ax.set_title('Top 20 CTR Review Candidates (Test Set)', fontsize=14, fontweight='bold', pad=20)
plt.savefig(CHARTS_DIR / '10_action_queue_top20.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 10 saved")

top20.to_csv(OUTPUT_DIR / 'refresh_queue_sample.csv', index=False)

# === 18. SAVE RESULTS JSON ===
model_results = {
    "research_question": "Which content signals distinguish high-CTR from low-CTR pages at similar search positions?",
    "dataset": "content_refresh_anonymized.csv",
    "validation": "client-holdout",
    "results": {name: {"roc_auc": round(res['auc'], 4), "avg_precision": round(res['ap'], 4), 
                       "precision_at_50": round(res['p50'], 4)} for name, res in results.items()}
}
with open(OUTPUT_DIR / 'model_results.json', 'w') as f:
    json.dump(model_results, f, indent=2)

print("\n=== ALL DONE ===")
print(f"Charts saved to: {CHARTS_DIR}")
print(f"Results saved to: {OUTPUT_DIR / 'model_results.json'}")
print(f"Queue saved to: {OUTPUT_DIR / 'refresh_queue_sample.csv'}")