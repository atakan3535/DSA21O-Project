# The Relationship Between Gameplay Metrics and Win Rate: A Case Study on Fiora in League of Legends
Atakan Öztürk, DSA210 Term Project, Fall 2025-2026

# Motivation and Overview

Competitive online games like League of Legends (LoL) have become one of the richest sources of performance-related data in modern eSports. Every match generates extensive statistics that can help players identify strengths, weaknesses, and optimal strategies for improvement. Many teams pay astronomic wages to coaches in order to utilize these data to team's success. 

As a professional League of Legends player who primarily mains the champion Fiora, I aim to use this project as both a data science exercise and a self-improvement tool. My motivation is to analyze how certain in-game metrics correlate with match outcomes (win/loss), and to determine which performance aspects most strongly predict victory when playing Fiora.

Specifically, this project focuses on the following key variables:

- **Kills** – Number of enemy champions killed.  
- **Deaths** – Number of times the player is defeated.  
- **Vision Score** – Contribution to map awareness and vision control.  
- **First Tower** – Whether the player’s team secures the first turret.  
- **Creep Score (CS)** – Number of minions or monsters killed for gold and experience.

These metrics represent different aspects of gameplay: mechanical skill (kills/deaths), macro awareness (vision score, first tower), and efficiency (CS). The goal is to uncover how deviations in these metrics from the average player’s performance affect the win rate.

By understanding the statistical weight of each variable, I aim to pinpoint which areas of performance improvement (e.g., farming consistency or objective pressure) yield the highest impact on my success rate. This study bridges subjective gameplay experience with objective data-driven insights.

# Dataset Overview

**Dataset:** Self-collected match history data from the Riot Games API
**Scope:** ~200+ ranked games played with Fiora in the top lane

| **Category** | **Variables** | **Description** |
|---------------|----------------|------------------|
| **Performance Metrics** | kills, deaths, assists | Core combat statistics |
| **Vision Metrics** | visionScore, wardsPlaced, wardsKilled | Map control contribution |
| **Objectives** | firstTower, damageToTurrets | Impact on early and mid-game objectives |
| **Economy** | totalMinionsKilled (CS), goldEarned | Resource efficiency and scaling |
| **Match Outcome** | win/loss | Target variable for correlation and regression |

Data will be collected via the Riot API and preprocessed for consistency. Outliers such as AFK games or remakes will be excluded to ensure clean analysis.

# Data Preparation & Cleaning

Data cleaning will involve:

- **Removing incomplete or erroneous matches** (e.g., AFK or remake games)  
- **Standardizing variables** (e.g., converting binary outcomes like “win/loss” to 1/0)  
- **Normalizing continuous values** (e.g., kills, CS) for fair comparison across games  
- **Calculating relative performance ratios** against the average statistics of Fiora players from public databases (e.g., League of Graphs, OP.GG)

# Exploratory Data Analysis (EDA)

The EDA phase will include:

- **Distribution plots** for kills, deaths, and CS  
- **Correlation matrix** between all metrics and win rate  
- **Comparative boxplots** between wins and losses for each metric  
- **Scatter plots** to visualize how CS or vision score influence win probability

**Libraries:** pandas, matplotlib, seaborn, numpy

# Hypothesis Testing

Example hypotheses include:

| **Null Hypothesis (H₀)** | **Alternative Hypothesis (H₁)** |
|----------------------------|----------------------------------|
| There is no relationship between CS and win rate. | Higher CS values are associated with higher win rates. |
| Vision score has no effect on match outcome. | Above-average vision score increases win probability. |
| First tower has no effect on victory likelihood. | Securing first tower significantly increases the chance of winning. |

# Expected Findings

- **Higher CS** and **lower death counts** are expected to correlate with higher win rates.  
- **Securing early objectives** like the first tower should show a strong positive correlation with winning.  
- **Vision score** may show moderate influence — valuable in higher ranks but less impactful in solo queue games.

# Limitations

- External factors such as matchmaking variance, teammate performance, or patch changes were not controlled for.

# Future Works

- **Incorporate** team-level statistics (e.g., overall gold difference, objective control).   
- **Expand** the dataset using Riot API to include multiple seasons or ranked tiers.  
