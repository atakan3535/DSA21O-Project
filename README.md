## Data Preparation
The script performs these steps:
1. Load `data/full_dataset.csv`
2. Filter only **Fiora TOP** matches
3. Create:
   - `win_numeric` (True/False → 1/0)
   - `kda` = (kills_14 + assists_14) / (deaths_14 + 1)

## Exploratory Data Analysis (EDA)
The EDA compares winners vs losers and generates plots.

### Summary Table (Winners vs Losers)
From the current run:

| Metric | Winners (mean) | Losers (mean) | Difference |
|---|---:|---:|---:|
| Kills (kills_14) | 1.94 | 1.24 | +0.70 |
| Deaths (deaths_14) | 1.83 | 2.55 | -0.72 |
| Gold (gold_14) | 4993.08 | 4479.22 | +513.86 |
| Plates (plates_14) | 6.72 | 4.00 | +2.72 |

### Visualizations (saved to `figures/`)
1. **Gold distribution** (wins vs losses)  
2. **Boxplots** for kills/deaths/gold/plates  
3. **Correlation heatmap** (including `win_numeric`)  
4. **Win rate by turret plates**  
5. **Confusion matrix** for the ML model  

## Hypothesis Testing
We test whether winners and losers differ in key early-game metrics using **independent t-tests** with significance level **α = 0.05**.

From the current run:

| Metric | t-stat | p-value | Result |
|---|---:|---:|---|
| Gold | 8.124 | < 0.001 | Significant |
| Deaths | -5.787 | < 0.001 | Significant |
| Plates | 13.364 | < 0.001 | Significant |
| KDA | 7.613 | < 0.001 | Significant |

Interpretation:
- Winners have **more gold**, **fewer deaths**, **more plates**, and **higher KDA** by 14 minutes.

## Machine Learning: Logistic Regression
A **logistic regression** model predicts win/loss from early-game features.

**Features used:**
- kills_14, deaths_14, assists_14, gold_14, plates_14

**Train/test split:** 80% / 20% (`random_state=42`)

From the current run:
- Training samples: **466**
- Testing samples: **117**
- Accuracy: **65.0%**

Feature coefficients (current run):
- kills_14: **+0.2585**
- deaths_14: **-0.3595**
- assists_14: **+0.1033**
- gold_14: **-0.0004**
- plates_14: **+0.4685**

Main takeaway:
- **plates_14** has the strongest positive impact in the model.
- **deaths_14** strongly reduces win probability.

## Key Findings (Current Run)
- Winners have about **+514 more gold at 14 minutes**
- Winners have about **0.7 fewer deaths**
- Winners take about **+2.7 more turret plates**
- These differences are **statistically significant**
- Logistic Regression predicts win/loss with **~65% accuracy**
- The strongest early-game signal is **turret plates**

## Recommendations (Gameplay)
Based on the analysis:
- Focus on taking **turret plates** early (better objective pressure).
- Minimize **deaths before 14 minutes** (avoid giving tempo + gold).
- Build a steady **gold lead** through farming and smart trades.

## How to Run
### 1) Install dependencies
Recommended (venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy matplotlib seaborn scipy scikit-learn
