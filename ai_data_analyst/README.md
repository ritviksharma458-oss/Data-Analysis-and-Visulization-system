# AI Data Analyst System

A complete AI-powered data analysis system built in **pure Python** — no web frameworks, no Streamlit. Runs entirely in your terminal.

---

## Features

- **Load any dataset** — CSV, Excel, JSON
- **18+ interactive graph types** — Line, Bar, Scatter, Heatmap, 3D, Animated, Pie, Violin, Sunburst, and more
- **Save graphs** — PNG, JPG, PDF, or interactive HTML
- **ML model training** — Regression, Classification, Clustering
- **Auto problem-type detection** — detects whether your target needs regression or classification
- **All models compared** — trains all relevant models and picks the best one
- **Accuracy control** — Fast / Balanced / High Accuracy modes
- **Clean terminal UI** — colour-coded, menu-driven, never crashes

---

## Installation

```bash
# 1. Clone or copy the project files
cd ai_analyst

# 2. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

```bash
python main.py
```

---

## Menu Options

| Option | Action |
|--------|--------|
| 1 | Load / Change Dataset |
| 2 | Show Data Preview & Summary |
| 3 | Generate Graph (18 types) |
| 4 | Train ML Models |
| 5 | Save Last Graph |
| 6 | Descriptive Statistics |
| 7 | Export Clean Data (CSV) |
| 0 | Exit |

---

## Example Usage

```
python main.py

[1] Load Dataset
Enter file path: sample.csv

[3] Generate Graph
Choose graph type: 9   ← Correlation Heatmap

[4] Train ML Models
Target column: price
Accuracy mode: 2 (Balanced)
→ Best model: Random Forest Regressor (R²=0.9231)

[5] Save Graph
Format: 1 (HTML)
→ Saved to outputs/graphs/heatmap_20240101_120000.html
```

---

## Project Structure

```
ai_analyst/
├── main.py            ← Entry point / CLI loop
├── utils.py           ← Helpers, column detection, UI
├── visualization.py   ← All 18 graph types (Plotly + Matplotlib)
├── models.py          ← ML training, evaluation, recommendation
├── requirements.txt   ← Dependencies
├── README.md          ← This file
└── outputs/
    ├── graphs/        ← Saved plots
    └── models/        ← Saved .pkl model files
```

---

## Supported Graph Types

1. Line Chart
2. Bar Chart (Vertical)
3. Bar Chart (Horizontal)
4. Scatter Plot
5. Histogram
6. Box Plot
7. Violin Plot
8. Pie Chart
9. Heatmap (Correlation Matrix)
10. Area Chart
11. Bubble Chart
12. 3D Scatter Plot
13. 3D Surface Plot
14. Animated Line Chart
15. Animated Bar Chart Race
16. Pair Plot
17. Distribution Plot (KDE)
18. Sunburst Chart

---

## ML Models Supported

**Regression:** Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting

**Classification:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting

**Clustering:** KMeans (auto best-K selection)

---

## Notes

- Graphs open in your **default browser** (interactive Plotly)
- Saved files go to `outputs/graphs/` and `outputs/models/`
- If `kaleido` is not installed, PNG/PDF saves fall back to HTML
