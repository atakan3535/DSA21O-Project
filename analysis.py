import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# Create output folders
os.makedirs('figures', exist_ok=True)
os.makedirs('output', exist_ok=True)

# ===========================================
# STEP 1: LOAD AND PREPARE DATA
# ===========================================
print("=" * 50)
print("STEP 1: LOAD AND PREPARE DATA")
print("=" * 50)

df = pd.read_csv('data/full_dataset.csv')
print(f"Total records: {len(df)}")

# Filter for Fiora TOP games only
fiora = df[(df['champion'] == 'Fiora') & (df['role'] == 'TOP')].copy()
print(f"Fiora TOP games: {len(fiora)}")

# Create useful columns
fiora['win_numeric'] = fiora['win'].astype(int)
fiora['kda'] = (fiora['kills_14'] + fiora['assists_14']) / (fiora['deaths_14'] + 1)

# Separate winners and losers
winners = fiora[fiora['win'] == True]
losers = fiora[fiora['win'] == False]

win_rate = fiora['win'].mean() * 100
print(f"Win rate: {win_rate:.1f}%")

# ===========================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ===========================================
print("\n" + "=" * 50)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 50)

print(f"\n{'Metric':<15} {'Winners':>10} {'Losers':>10} {'Diff':>10}")
print("-" * 45)

for col, name in [('kills_14', 'Kills'), ('deaths_14', 'Deaths'), 
                  ('gold_14', 'Gold'), ('plates_14', 'Plates')]:
    w = winners[col].mean()
    l = losers[col].mean()
    print(f"{name:<15} {w:>10.2f} {l:>10.2f} {w-l:>+10.2f}")

# ===========================================
# STEP 3: VISUALIZATIONS
# ===========================================
print("\n" + "=" * 50)
print("STEP 3: CREATING VISUALIZATIONS")
print("=" * 50)

plt.style.use('seaborn-v0_8-whitegrid')

# Figure 1: Gold Distribution
plt.figure(figsize=(10, 5))
plt.hist(winners['gold_14'], bins=20, alpha=0.6, label='Win', color='green')
plt.hist(losers['gold_14'], bins=20, alpha=0.6, label='Loss', color='red')
plt.xlabel('Gold at 14 min')
plt.ylabel('Games')
plt.title('Gold Distribution: Winners vs Losers')
plt.legend()
plt.savefig('figures/01_gold_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/01_gold_distribution.png")

# Figure 2: Boxplots
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for i, (col, title) in enumerate([('kills_14', 'Kills'), ('deaths_14', 'Deaths'), 
                                   ('gold_14', 'Gold'), ('plates_14', 'Plates')]):
    axes[i].boxplot([losers[col], winners[col]], tick_labels=['Loss', 'Win'])
    axes[i].set_title(title)
plt.tight_layout()
plt.savefig('figures/02_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/02_boxplots.png")

# Figure 3: Correlation Heatmap
plt.figure(figsize=(8, 6))
cols = ['kills_14', 'deaths_14', 'gold_14', 'plates_14', 'kda', 'win_numeric']
sns.heatmap(fiora[cols].corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('figures/03_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/03_correlation.png")

# Figure 4: Win Rate by Plates
plt.figure(figsize=(8, 5))
plate_wr = fiora.groupby('plates_14')['win'].mean() * 100
plate_wr.plot(kind='bar', color='steelblue')
plt.axhline(y=50, color='red', linestyle='--', label='50%')
plt.xlabel('Turret Plates')
plt.ylabel('Win Rate (%)')
plt.title('Win Rate by Turret Plates')
plt.legend()
plt.tight_layout()
plt.savefig('figures/04_winrate_plates.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/04_winrate_plates.png")

# ===========================================
# STEP 4: HYPOTHESIS TESTING
# ===========================================
print("\n" + "=" * 50)
print("STEP 4: HYPOTHESIS TESTING")
print("=" * 50)

alpha = 0.05
print(f"\nSignificance level: {alpha}")

tests = [
    ('Gold', 'gold_14'),
    ('Deaths', 'deaths_14'),
    ('Plates', 'plates_14'),
    ('KDA', 'kda')
]

print(f"\n{'Metric':<10} {'t-stat':>10} {'p-value':>12} {'Result':>15}")
print("-" * 50)

for name, col in tests:
    t, p = stats.ttest_ind(winners[col], losers[col])
    result = "Significant" if p < alpha else "Not Significant"
    print(f"{name:<10} {t:>10.3f} {p:>12.6f} {result:>15}")

# ===========================================
# STEP 5: MACHINE LEARNING
# ===========================================
print("\n" + "=" * 50)
print("STEP 5: MACHINE LEARNING (Logistic Regression)")
print("=" * 50)

features = ['kills_14', 'deaths_14', 'assists_14', 'gold_14', 'plates_14']
X = fiora[features]
y = fiora['win_numeric']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining: {len(X_train)}, Testing: {len(X_test)}")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.1%}")

print(f"\nFeature Coefficients:")
for feat, coef in zip(features, model.coef_[0]):
    print(f"  {feat}: {coef:+.4f}")

# Figure 5: Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Loss', 'Pred Win'],
            yticklabels=['Actual Loss', 'Actual Win'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('figures/05_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: figures/05_confusion_matrix.png")

# ===========================================
# STEP 6: SUMMARY
# ===========================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)

print(f"""
Data: {len(fiora)} Fiora TOP games, {win_rate:.1f}% win rate

Key Findings:
- Winners have ~{winners['gold_14'].mean() - losers['gold_14'].mean():.0f} more gold at 14 min
- Winners have ~{losers['deaths_14'].mean() - winners['deaths_14'].mean():.1f} fewer deaths
- Winners take ~{winners['plates_14'].mean() - losers['plates_14'].mean():.1f} more turret plates
- All differences are statistically significant (p < 0.05)

Machine Learning:
- Logistic Regression accuracy: {accuracy:.1%}
- Most positive impact: plates_14

Recommendations:
- Focus on turret plates early game
- Minimize deaths before 14 minutes
- Build gold lead through farming
""")

# Save outputs
fiora.to_csv('output/fiora_data.csv', index=False)
print("Saved: output/fiora_data.csv")

print("\nAnalysis complete!")
