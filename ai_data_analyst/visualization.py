"""
visualization.py — Complete graph system using Plotly + Matplotlib fallback.
Supports 15+ chart types, animated graphs, and save/export functionality.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Plotly imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
    pio.renderers.default = "browser"
except ImportError:
    PLOTLY_AVAILABLE = False

# Matplotlib imports
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    print_success, print_error, print_warning, print_info,
    print_section, bold, green, cyan, yellow, dim,
    timestamped_filename, ensure_output_dir, prompt,
    pick_column, pick_multiple_columns, detect_column_types, press_enter
)


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_plotly(fig, name: str, fmt: str, out_dir: Path) -> str:
    filename = timestamped_filename(name, fmt)
    path     = out_dir / filename

    if fmt == "html":
        fig.write_html(str(path))
    elif fmt in ("png", "jpg", "jpeg", "pdf"):
        try:
            fig.write_image(str(path))
        except Exception:
            # Fallback: save as HTML if kaleido not installed
            path = out_dir / timestamped_filename(name, "html")
            fig.write_html(str(path))
            print_warning("kaleido not installed — saved as HTML instead.")
    else:
        fig.write_html(str(path))

    print_success(f"Graph saved → {path}")
    return str(path)


def _save_matplotlib(fig, name: str, fmt: str, out_dir: Path) -> str:
    filename = timestamped_filename(name, fmt)
    path     = out_dir / filename
    fig.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print_success(f"Graph saved → {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Graph registry
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_MENU = {
    "1":  "Line Chart",
    "2":  "Bar Chart (Vertical)",
    "3":  "Bar Chart (Horizontal)",
    "4":  "Scatter Plot",
    "5":  "Histogram",
    "6":  "Box Plot",
    "7":  "Violin Plot",
    "8":  "Pie Chart",
    "9":  "Heatmap (Correlation Matrix)",
    "10": "Area Chart",
    "11": "Bubble Chart",
    "12": "3D Scatter Plot",
    "13": "3D Surface Plot",
    "14": "Animated Line Chart",
    "15": "Animated Bar Chart Race",
    "16": "Pair Plot",
    "17": "Distribution Plot (KDE)",
    "18": "Sunburst Chart",
}


def print_graph_menu():
    print_section("GRAPH MENU")
    for k, v in GRAPH_MENU.items():
        print(f"  {dim(k):>6}. {v}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual graph builders
# ─────────────────────────────────────────────────────────────────────────────

class GraphBuilder:
    """Builds all supported graph types from a DataFrame."""

    def __init__(self, df: pd.DataFrame, out_dir: str = "outputs/graphs"):
        self.df      = df
        self.out_dir = ensure_output_dir(out_dir)
        self.col_types = detect_column_types(df)
        self._last_fig = None   # store last plotly figure for saving

    # ------------------------------------------------------------------ #
    # 1. Line Chart
    # ------------------------------------------------------------------ #
    def line_chart(self) -> None:
        x_col = pick_column(self.df, "X-axis column")
        if not x_col:
            return
        y_cols = pick_multiple_columns(
            self.df, "Y-axis column(s)", allowed_types=["numeric"]
        )
        if not y_cols:
            return

        title = prompt("Chart title", f"Line Chart — {', '.join(y_cols)} vs {x_col}")

        if PLOTLY_AVAILABLE:
            fig = px.line(self.df, x=x_col, y=y_cols, title=title,
                          template="plotly_dark",
                          labels={c: c for c in y_cols})
            fig.update_layout(hovermode="x unified")
            fig.show()
            self._last_fig = (fig, "line_chart", "plotly")
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            for col in y_cols:
                ax.plot(self.df[x_col], self.df[col], label=col, linewidth=2)
            ax.set_title(title); ax.set_xlabel(x_col); ax.legend()
            plt.tight_layout(); plt.show()
            self._last_fig = (fig, "line_chart", "matplotlib")

    # ------------------------------------------------------------------ #
    # 2 & 3. Bar Charts
    # ------------------------------------------------------------------ #
    def bar_chart(self, orientation: str = "v") -> None:
        x_col = pick_column(self.df, "Category column (X-axis)")
        if not x_col:
            return
        y_col = pick_column(self.df, "Value column (Y-axis)",
                            allowed_types=["numeric"])
        if not y_col:
            return

        agg_df = self.df.groupby(x_col)[y_col].mean().reset_index()
        title  = prompt("Chart title", f"Bar Chart — {y_col} by {x_col}")

        if PLOTLY_AVAILABLE:
            if orientation == "v":
                fig = px.bar(agg_df, x=x_col, y=y_col, title=title,
                             template="plotly_dark", color=y_col,
                             color_continuous_scale="Viridis")
            else:
                fig = px.bar(agg_df, x=y_col, y=x_col, title=title,
                             orientation="h", template="plotly_dark",
                             color=y_col, color_continuous_scale="Viridis")
            fig.show()
            self._last_fig = (fig, "bar_chart", "plotly")
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            if orientation == "v":
                ax.bar(agg_df[x_col].astype(str), agg_df[y_col],
                       color=sns.color_palette("viridis", len(agg_df)))
                ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                plt.xticks(rotation=45, ha="right")
            else:
                ax.barh(agg_df[x_col].astype(str), agg_df[y_col],
                        color=sns.color_palette("viridis", len(agg_df)))
                ax.set_xlabel(y_col); ax.set_ylabel(x_col)
            ax.set_title(title); plt.tight_layout(); plt.show()
            self._last_fig = (fig, "bar_chart", "matplotlib")

    # ------------------------------------------------------------------ #
    # 4. Scatter Plot
    # ------------------------------------------------------------------ #
    def scatter_plot(self) -> None:
        x_col = pick_column(self.df, "X-axis column", allowed_types=["numeric"])
        if not x_col:
            return
        y_col = pick_column(self.df, "Y-axis column", allowed_types=["numeric"])
        if not y_col:
            return

        color_col = None
        if self.col_types["categorical"]:
            use_color = prompt(
                f"Color by category? ({', '.join(self.col_types['categorical'][:5])}) "
                f"or press Enter to skip"
            )
            if use_color in self.df.columns:
                color_col = use_color

        title = prompt("Chart title", f"Scatter Plot — {y_col} vs {x_col}")

        if PLOTLY_AVAILABLE:
            fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col,
                             title=title, template="plotly_dark",
                             trendline="ols" if color_col is None else None,
                             hover_data=self.df.columns.tolist()[:5])
            fig.show()
            self._last_fig = (fig, "scatter_plot", "plotly")
        else:
            fig, ax = plt.subplots(figsize=(10, 7))
            if color_col:
                groups = self.df.groupby(color_col)
                for name, grp in groups:
                    ax.scatter(grp[x_col], grp[y_col], label=name, alpha=0.7, s=50)
                ax.legend()
            else:
                ax.scatter(self.df[x_col], self.df[y_col], alpha=0.6, s=40,
                           color="#4C72B0")
            ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(title)
            plt.tight_layout(); plt.show()
            self._last_fig = (fig, "scatter_plot", "matplotlib")

    # ------------------------------------------------------------------ #
    # 5. Histogram
    # ------------------------------------------------------------------ #
    def histogram(self) -> None:
        col = pick_column(self.df, "Column to plot", allowed_types=["numeric"])
        if not col:
            return
        bins = int(prompt("Number of bins", "30"))
        title = prompt("Chart title", f"Histogram — {col}")

        if PLOTLY_AVAILABLE:
            fig = px.histogram(self.df, x=col, nbins=bins, title=title,
                               template="plotly_dark",
                               marginal="box",
                               color_discrete_sequence=["#00B4D8"])
            fig.show()
            self._last_fig = (fig, "histogram", "plotly")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].hist(self.df[col].dropna(), bins=bins,
                         color="#4C72B0", edgecolor="white", alpha=0.85)
            axes[0].set_title(title); axes[0].set_xlabel(col)
            sns.boxplot(y=self.df[col].dropna(), ax=axes[1], color="#55A868")
            axes[1].set_title(f"Box Plot — {col}")
            plt.tight_layout(); plt.show()
            self._last_fig = (fig, "histogram", "matplotlib")

    # ------------------------------------------------------------------ #
    # 6. Box Plot
    # ------------------------------------------------------------------ #
    def box_plot(self) -> None:
        y_cols = pick_multiple_columns(
            self.df, "Columns to plot", allowed_types=["numeric"]
        )
        if not y_cols:
            return

        cat_col = None
        if self.col_types["categorical"]:
            raw = prompt(
                f"Group by category? ({', '.join(self.col_types['categorical'][:5])}) "
                f"or press Enter to skip"
            )
            if raw in self.df.columns:
                cat_col = raw

        title = prompt("Chart title", "Box Plot")

        if PLOTLY_AVAILABLE:
            if cat_col:
                fig = px.box(self.df, x=cat_col, y=y_cols[0], title=title,
                             template="plotly_dark", color=cat_col,
                             points="outliers")
            else:
                fig = go.Figure()
                for col in y_cols:
                    fig.add_trace(go.Box(y=self.df[col].dropna(),
                                         name=col, boxpoints="outliers"))
                fig.update_layout(title=title, template="plotly_dark")
            fig.show()
            self._last_fig = (fig, "box_plot", "plotly")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            data = [self.df[c].dropna().values for c in y_cols]
            ax.boxplot(data, labels=y_cols, patch_artist=True,
                       boxprops=dict(facecolor="#4C72B0", alpha=0.7))
            ax.set_title(title); plt.xticks(rotation=30)
            plt.tight_layout(); plt.show()
            self._last_fig = (fig, "box_plot", "matplotlib")

    # ------------------------------------------------------------------ #
    # 7. Violin Plot
    # ------------------------------------------------------------------ #
    def violin_plot(self) -> None:
        y_col = pick_column(self.df, "Value column", allowed_types=["numeric"])
        if not y_col:
            return

        cat_col = None
        if self.col_types["categorical"]:
            raw = prompt(
                f"Group by? ({', '.join(self.col_types['categorical'][:5])}) "
                f"or Enter to skip"
            )
            if raw in self.df.columns:
                cat_col = raw

        title = prompt("Chart title", f"Violin Plot — {y_col}")

        if PLOTLY_AVAILABLE:
            fig = px.violin(self.df, y=y_col, x=cat_col, title=title,
                            template="plotly_dark", box=True, points="all",
                            color=cat_col)
            fig.show()
            self._last_fig = (fig, "violin_plot", "plotly")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            if cat_col:
                groups  = sorted(self.df[cat_col].dropna().unique())
                data    = [self.df[self.df[cat_col] == g][y_col].dropna()
                           for g in groups]
                parts   = ax.violinplot(data, showmedians=True)
                ax.set_xticks(range(1, len(groups) + 1))
                ax.set_xticklabels(groups, rotation=30)
            else:
                ax.violinplot(self.df[y_col].dropna(), showmedians=True)
            ax.set_title(title); ax.set_ylabel(y_col)
            plt.tight_layout(); plt.show()
            self._last_fig = (fig, "violin_plot", "matplotlib")

    # ------------------------------------------------------------------ #
    # 8. Pie Chart
    # ------------------------------------------------------------------ #
    def pie_chart(self) -> None:
        col = pick_column(self.df, "Category column",
                          allowed_types=["categorical"])
        if not col:
            col = pick_column(self.df, "Column for pie slices")
        if not col:
            return

        val_col = pick_column(self.df, "Value column (or Enter to use counts)",
                              allowed_types=["numeric"])
        title   = prompt("Chart title", f"Pie Chart — {col}")

        if val_col:
            pie_df = self.df.groupby(col)[val_col].sum().reset_index()
            values, names = pie_df[val_col], pie_df[col]
        else:
            counts = self.df[col].value_counts().reset_index()
            counts.columns = [col, "count"]
            values, names = counts["count"], counts[col]

        if PLOTLY_AVAILABLE:
            fig = px.pie(values=values, names=names, title=title,
                         template="plotly_dark", hole=0.35)
            fig.update_traces(textposition="inside",
                              textinfo="percent+label")
            fig.show()
            self._last_fig = (fig, "pie_chart", "plotly")
        else:
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.pie(values, labels=names, autopct="%1.1f%%",
                   startangle=140,
                   colors=sns.color_palette("pastel", len(values)))
            ax.set_title(title)
            plt.tight_layout(); plt.show()
            self._last_fig = (fig, "pie_chart", "matplotlib")

    # ------------------------------------------------------------------ #
    # 9. Heatmap / Correlation Matrix
    # ------------------------------------------------------------------ #
    def heatmap(self) -> None:
        num_df = self.df.select_dtypes(include=np.number)
        if num_df.shape[1] < 2:
            print_warning("Need at least 2 numeric columns.")
            return

        title = prompt("Chart title", "Correlation Matrix Heatmap")
        corr  = num_df.corr()

        if PLOTLY_AVAILABLE:
            fig = px.imshow(corr, text_auto=".2f", title=title,
                            color_continuous_scale="RdBu_r",
                            aspect="auto",
                            template="plotly_dark")
            fig.update_layout(width=800, height=700)
            fig.show()
            self._last_fig = (fig, "heatmap", "plotly")
        else:
            mask = np.triu(np.ones_like(corr, dtype=bool))
            fig, ax = plt.subplots(
                figsize=(max(8, len(corr) * 0.8), max(6, len(corr) * 0.7))
            )
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                        cmap="coolwarm", center=0,
                        linewidths=0.5, ax=ax)
            ax.set_title(title)
            plt.tight_layout(); plt.show()
            self._last_fig = (fig, "heatmap", "matplotlib")

    # ------------------------------------------------------------------ #
    # 10. Area Chart
    # ------------------------------------------------------------------ #
    def area_chart(self) -> None:
        x_col  = pick_column(self.df, "X-axis column")
        if not x_col:
            return
        y_cols = pick_multiple_columns(
            self.df, "Y-axis column(s)", allowed_types=["numeric"]
        )
        if not y_cols:
            return
        title = prompt("Chart title", f"Area Chart — {', '.join(y_cols)}")

        if PLOTLY_AVAILABLE:
            fig = px.area(self.df, x=x_col, y=y_cols, title=title,
                          template="plotly_dark")
            fig.show()
            self._last_fig = (fig, "area_chart", "plotly")
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            for col in y_cols:
                ax.fill_between(range(len(self.df)), self.df[col],
                                label=col, alpha=0.5)
                ax.plot(self.df[col], linewidth=1.5)
            ax.set_title(title); ax.legend()
            plt.tight_layout(); plt.show()
            self._last_fig = (fig, "area_chart", "matplotlib")

    # ------------------------------------------------------------------ #
    # 11. Bubble Chart
    # ------------------------------------------------------------------ #
    def bubble_chart(self) -> None:
        x_col    = pick_column(self.df, "X-axis column", allowed_types=["numeric"])
        y_col    = pick_column(self.df, "Y-axis column", allowed_types=["numeric"])
        size_col = pick_column(self.df, "Bubble size column", allowed_types=["numeric"])
        if not all([x_col, y_col, size_col]):
            return

        color_col = None
        if self.col_types["categorical"]:
            raw = prompt(f"Color by? ({', '.join(self.col_types['categorical'][:4])}) or Enter")
            if raw in self.df.columns:
                color_col = raw

        title = prompt("Chart title", f"Bubble Chart — {y_col} vs {x_col}")

        if PLOTLY_AVAILABLE:
            fig = px.scatter(self.df, x=x_col, y=y_col,
                             size=size_col, color=color_col,
                             title=title, template="plotly_dark",
                             size_max=60, hover_name=color_col)
            fig.show()
            self._last_fig = (fig, "bubble_chart", "plotly")
        else:
            sizes = (self.df[size_col] - self.df[size_col].min()) / \
                    (self.df[size_col].max() - self.df[size_col].min()) * 500 + 20
            fig, ax = plt.subplots(figsize=(10, 7))
            scatter = ax.scatter(self.df[x_col], self.df[y_col],
                                 s=sizes, alpha=0.6, c=sizes, cmap="viridis")
            plt.colorbar(scatter, ax=ax, label=size_col)
            ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(title)
            plt.tight_layout(); plt.show()
            self._last_fig = (fig, "bubble_chart", "matplotlib")

    # ------------------------------------------------------------------ #
    # 12. 3D Scatter Plot
    # ------------------------------------------------------------------ #
    def scatter_3d(self) -> None:
        if not PLOTLY_AVAILABLE:
            print_warning("3D Scatter requires Plotly.")
            return

        x_col = pick_column(self.df, "X-axis column", allowed_types=["numeric"])
        y_col = pick_column(self.df, "Y-axis column", allowed_types=["numeric"])
        z_col = pick_column(self.df, "Z-axis column", allowed_types=["numeric"])
        if not all([x_col, y_col, z_col]):
            return

        color_col = None
        if self.col_types["categorical"]:
            raw = prompt(f"Color by? or Enter to skip")
            if raw in self.df.columns:
                color_col = raw

        title = prompt("Chart title", f"3D Scatter — {x_col}, {y_col}, {z_col}")
        fig   = px.scatter_3d(self.df, x=x_col, y=y_col, z=z_col,
                               color=color_col, title=title,
                               template="plotly_dark", opacity=0.75)
        fig.show()
        self._last_fig = (fig, "scatter_3d", "plotly")

    # ------------------------------------------------------------------ #
    # 13. 3D Surface Plot
    # ------------------------------------------------------------------ #
    def surface_3d(self) -> None:
        num_cols = self.col_types["numeric"]
        if len(num_cols) < 3:
            print_warning("Need at least 3 numeric columns for a surface plot.")
            return

        if not PLOTLY_AVAILABLE:
            print_warning("3D Surface requires Plotly.")
            return

        print_info("Using first 3 numeric columns for X, Y, Z.")
        x_col, y_col, z_col = num_cols[:3]

        try:
            xi = np.linspace(self.df[x_col].min(), self.df[x_col].max(), 40)
            yi = np.linspace(self.df[y_col].min(), self.df[y_col].max(), 40)
            xi, yi = np.meshgrid(xi, yi)
            from scipy.interpolate import griddata
            zi = griddata(
                (self.df[x_col], self.df[y_col]),
                self.df[z_col],
                (xi, yi), method="linear"
            )
        except ImportError:
            zi = np.outer(
                np.linspace(0, 1, 40),
                np.linspace(0, 1, 40)
            ) * self.df[z_col].mean()

        title = prompt("Chart title", f"3D Surface — {z_col}")
        fig   = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi,
                                            colorscale="Viridis")])
        fig.update_layout(title=title, template="plotly_dark",
                          scene=dict(xaxis_title=x_col,
                                     yaxis_title=y_col,
                                     zaxis_title=z_col))
        fig.show()
        self._last_fig = (fig, "surface_3d", "plotly")

    # ------------------------------------------------------------------ #
    # 14. Animated Line Chart
    # ------------------------------------------------------------------ #
    def animated_line(self) -> None:
        if not PLOTLY_AVAILABLE:
            print_warning("Animated charts require Plotly.")
            return

        dt_cols = self.col_types["datetime"]
        if dt_cols:
            time_col = dt_cols[0]
            print_info(f"Using detected datetime column: {time_col}")
        else:
            time_col = pick_column(self.df, "Time / animation column")
        if not time_col:
            return

        y_col    = pick_column(self.df, "Y-axis column", allowed_types=["numeric"])
        cat_col  = None
        if self.col_types["categorical"]:
            raw = prompt("Color / group by? or Enter to skip")
            if raw in self.df.columns:
                cat_col = raw

        title = prompt("Chart title", f"Animated Line — {y_col} over {time_col}")
        fig   = px.line(self.df.sort_values(time_col),
                        x=time_col, y=y_col, color=cat_col,
                        title=title, template="plotly_dark",
                        animation_frame=time_col if self.df[time_col].nunique() <= 100 else None)
        fig.show()
        self._last_fig = (fig, "animated_line", "plotly")

    # ------------------------------------------------------------------ #
    # 15. Animated Bar Chart Race
    # ------------------------------------------------------------------ #
    def animated_bar_race(self) -> None:
        if not PLOTLY_AVAILABLE:
            print_warning("Animated charts require Plotly.")
            return

        dt_cols  = self.col_types["datetime"]
        time_col = dt_cols[0] if dt_cols else pick_column(
            self.df, "Time / frame column"
        )
        cat_col  = pick_column(self.df, "Category column",
                               allowed_types=["categorical"])
        val_col  = pick_column(self.df, "Value column",
                               allowed_types=["numeric"])
        if not all([time_col, cat_col, val_col]):
            return

        title = prompt("Chart title", f"Animated Bar Race — {val_col} by {cat_col}")
        fig   = px.bar(self.df.sort_values(time_col),
                       x=val_col, y=cat_col, color=cat_col,
                       animation_frame=time_col,
                       animation_group=cat_col,
                       orientation="h", title=title,
                       template="plotly_dark",
                       range_x=[0, self.df[val_col].max() * 1.1])
        fig.show()
        self._last_fig = (fig, "animated_bar_race", "plotly")

    # ------------------------------------------------------------------ #
    # 16. Pair Plot
    # ------------------------------------------------------------------ #
    def pair_plot(self) -> None:
        num_cols = self.col_types["numeric"]
        if len(num_cols) < 2:
            print_warning("Need at least 2 numeric columns.")
            return

        cols = num_cols[:6]
        print_info(f"Using columns: {cols}")

        color_col = None
        if self.col_types["categorical"]:
            raw = prompt(f"Color by? or Enter to skip")
            if raw in self.df.columns:
                color_col = raw

        title = prompt("Chart title", "Pair Plot")

        if PLOTLY_AVAILABLE:
            fig = px.scatter_matrix(self.df, dimensions=cols,
                                     color=color_col, title=title,
                                     template="plotly_dark")
            fig.update_traces(diagonal_visible=True,
                              showupperhalf=False)
            fig.show()
            self._last_fig = (fig, "pair_plot", "plotly")
        else:
            g = sns.pairplot(self.df[cols + ([color_col] if color_col else [])].dropna(),
                             hue=color_col, diag_kind="kde",
                             plot_kws=dict(alpha=0.5, s=20))
            g.fig.suptitle(title, y=1.01)
            self._last_fig = (g.fig, "pair_plot", "matplotlib")

    # ------------------------------------------------------------------ #
    # 17. Distribution / KDE Plot
    # ------------------------------------------------------------------ #
    def distribution_plot(self) -> None:
        cols = pick_multiple_columns(
            self.df, "Column(s) to plot", allowed_types=["numeric"]
        )
        if not cols:
            return
        title = prompt("Chart title", "Distribution Plot")

        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            for col in cols:
                fig.add_trace(
                    go.Histogram(x=self.df[col].dropna(), name=col,
                                 histnorm="probability density",
                                 opacity=0.65, nbinsx=40)
                )
            fig.update_layout(barmode="overlay", title=title,
                              template="plotly_dark",
                              xaxis_title="Value",
                              yaxis_title="Density")
            fig.show()
            self._last_fig = (fig, "distribution_plot", "plotly")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            for col in cols:
                self.df[col].dropna().plot.kde(ax=ax, label=col, linewidth=2)
            ax.set_title(title); ax.legend(); ax.set_xlabel("Value")
            plt.tight_layout(); plt.show()
            self._last_fig = (fig, "distribution_plot", "matplotlib")

    # ------------------------------------------------------------------ #
    # 18. Sunburst Chart
    # ------------------------------------------------------------------ #
    def sunburst(self) -> None:
        if not PLOTLY_AVAILABLE:
            print_warning("Sunburst chart requires Plotly.")
            return

        cat_cols = pick_multiple_columns(
            self.df, "Hierarchy columns (outer→inner)",
            allowed_types=["categorical"], min_count=1
        )
        val_col = pick_column(self.df, "Value column",
                              allowed_types=["numeric"])
        if not cat_cols:
            return

        title = prompt("Chart title", "Sunburst Chart")
        fig   = px.sunburst(self.df, path=cat_cols,
                             values=val_col, title=title,
                             template="plotly_dark",
                             color_continuous_scale="RdBu")
        fig.show()
        self._last_fig = (fig, "sunburst", "plotly")

    # ------------------------------------------------------------------ #
    # Save last graph
    # ------------------------------------------------------------------ #
    def save_last_graph(self) -> None:
        if self._last_fig is None:
            print_warning("No graph generated yet.")
            return

        fig, name, engine = self._last_fig

        print("\n  Save format:")
        print("    1. HTML  (interactive)")
        print("    2. PNG")
        print("    3. JPG")
        print("    4. PDF")
        fmt_map = {"1": "html", "2": "png", "3": "jpg", "4": "pdf"}
        choice  = prompt("Choose format", "1")
        fmt     = fmt_map.get(choice, "html")

        if engine == "plotly":
            _save_plotly(fig, name, fmt, self.out_dir)
        else:
            _save_matplotlib(fig, name, fmt if fmt != "html" else "png",
                             self.out_dir)

    # ------------------------------------------------------------------ #
    # Dispatcher
    # ------------------------------------------------------------------ #
    def build(self, choice: str) -> None:
        dispatch = {
            "1":  self.line_chart,
            "2":  self.bar_chart,
            "3":  lambda: self.bar_chart(orientation="h"),
            "4":  self.scatter_plot,
            "5":  self.histogram,
            "6":  self.box_plot,
            "7":  self.violin_plot,
            "8":  self.pie_chart,
            "9":  self.heatmap,
            "10": self.area_chart,
            "11": self.bubble_chart,
            "12": self.scatter_3d,
            "13": self.surface_3d,
            "14": self.animated_line,
            "15": self.animated_bar_race,
            "16": self.pair_plot,
            "17": self.distribution_plot,
            "18": self.sunburst,
        }
        fn = dispatch.get(choice)
        if fn:
            fn()
        else:
            print_warning("Invalid graph choice.")
