"""
EDA Utilities — Step 1.1
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from wordcloud import WordCloud
    _WC_OK = True
except ImportError:
    _WC_OK = False
    print("⚠️  pip install wordcloud")

# Hỗ trợ cả relative import (khi chạy như package)
# và absolute import (khi load trực tiếp qua importlib)
try:
    from .dataset import ENTITIES, ASPECT_CATEGORIES, SENTIMENTS
except ImportError:
    # Thêm thư mục src/data vào path để import trực tiếp
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from dataset import ENTITIES, ASPECT_CATEGORIES, SENTIMENTS

# EDA dùng entity-level (gộp attributes)
ASPECTS  = ENTITIES
_PALETTE = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#90A4AE"}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class EDAAnalyzer:
    def __init__(self, df: pd.DataFrame, save_dir: str = "outputs/eda_figures"):
        self.df  = df.copy()
        self.out = Path(save_dir)
        _ensure_dir(self.out)
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams["figure.dpi"] = 120

    # ------------------------------------------------------------------ #
    def summary(self) -> pd.DataFrame:
        rows = []
        for split, grp in self.df.groupby("split"):
            rows.append({
                "split":     split,
                "n_reviews": len(grp),
                "avg_chars": round(grp["text_len"].mean(), 1),
                "med_chars": round(grp["text_len"].median(), 1),
                "avg_words": round(grp["word_count"].mean(), 1),
                "max_words": int(grp["word_count"].max()),
            })
        stat = pd.DataFrame(rows)
        print(stat.to_string(index=False))
        return stat

    # ------------------------------------------------------------------ #
    def plot_text_length(self, save: bool = True) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, col, xlabel in zip(axes, ["text_len", "word_count"], ["Số ký tự", "Số từ"]):
            for split, grp in self.df.groupby("split"):
                ax.hist(grp[col], bins=40, alpha=0.55, label=split, edgecolor="none")
            ax.set_xlabel(xlabel); ax.set_ylabel("Số lượng")
            ax.set_title(f"Phân phối {xlabel}"); ax.legend()
        fig.suptitle("Thống kê độ dài văn bản — VLSP 2018", fontweight="bold")
        fig.tight_layout()
        if save:
            fig.savefig(self.out / "text_length_dist.png")
        plt.show()

    # ------------------------------------------------------------------ #
    def plot_aspect_distribution(self, save: bool = True) -> None:
        records = []
        for asp in ASPECTS:
            col = f"label_{asp.lower()}"
            if col not in self.df.columns:
                continue
            for sent, cnt in self.df[col].fillna("Missing").value_counts().items():
                records.append({"Aspect": asp, "Sentiment": sent, "Count": cnt})

        pivot = pd.DataFrame(records).pivot_table(
            index="Aspect", columns="Sentiment", values="Count", fill_value=0
        )
        color_map = {**_PALETTE, "Missing": "#E0E0E0"}
        ordered   = [c for c in ["positive", "negative", "neutral", "Missing"] if c in pivot.columns]
        ax = pivot[ordered].plot(kind="bar", stacked=True, figsize=(11, 5),
                                 color=[color_map[c] for c in ordered],
                                 edgecolor="white", linewidth=0.5)
        ax.set_title("Phân phối Entity × Sentiment — VLSP 2018", fontsize=13, fontweight="bold")
        ax.set_xlabel("Entity"); ax.set_ylabel("Số reviews")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.xticks(rotation=30, ha="right"); plt.tight_layout()
        if save:
            plt.savefig(self.out / "aspect_distribution.png")
        plt.show()

    # ------------------------------------------------------------------ #
    def plot_category_distribution(self, save: bool = True) -> None:
        """Biểu đồ chi tiết theo ENTITY#ATTRIBUTE."""
        records = []
        for cat in ASPECT_CATEGORIES:
            col = "cat_" + cat.lower().replace("#", "_")
            if col not in self.df.columns:
                continue
            for sent, cnt in self.df[col].fillna("Missing").value_counts().items():
                records.append({"Category": cat, "Sentiment": sent, "Count": cnt})

        if not records:
            print("⚠️ Không tìm thấy cột category trong DataFrame")
            return

        pivot = pd.DataFrame(records).pivot_table(
            index="Category", columns="Sentiment", values="Count", fill_value=0
        )
        color_map = {**_PALETTE, "Missing": "#E0E0E0"}
        ordered   = [c for c in ["positive", "negative", "neutral", "Missing"] if c in pivot.columns]
        ax = pivot[ordered].plot(kind="barh", stacked=True, figsize=(11, 8),
                                 color=[color_map[c] for c in ordered],
                                 edgecolor="white", linewidth=0.5)
        ax.set_title("Phân phối chi tiết ENTITY#ATTRIBUTE — VLSP 2018", fontsize=13, fontweight="bold")
        ax.set_xlabel("Số reviews"); ax.set_ylabel("Category")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()
        if save:
            plt.savefig(self.out / "category_distribution.png")
        plt.show()

    # ------------------------------------------------------------------ #
    def class_imbalance_report(self) -> pd.DataFrame:
        rows = []
        for asp in ASPECTS:
            col = f"label_{asp.lower()}"
            if col not in self.df.columns:
                continue
            total     = len(self.df)
            mentioned = self.df[col].notna().sum()
            asp_df    = self.df[self.df[col].notna()][col]
            rows.append({
                "entity":        asp,
                "mentioned":     int(mentioned),
                "not_mentioned": int(total - mentioned),
                "mention_ratio": round(mentioned / total, 3),
                "pos_ratio":     round((asp_df == "positive").sum() / max(mentioned, 1), 3),
                "neg_ratio":     round((asp_df == "negative").sum() / max(mentioned, 1), 3),
                "neu_ratio":     round((asp_df == "neutral").sum()  / max(mentioned, 1), 3),
            })
        report = pd.DataFrame(rows)
        print("\n=== CLASS IMBALANCE REPORT ===")
        print(report.to_string(index=False))
        return report

    # ------------------------------------------------------------------ #
    def plot_imbalance_heatmap(self, save: bool = True) -> None:
        matrix, valid = [], []
        for asp in ASPECTS:
            col = f"label_{asp.lower()}"
            if col not in self.df.columns:
                continue
            asp_df = self.df[self.df[col].notna()][col]
            total  = max(len(asp_df), 1)
            matrix.append([(asp_df == s).sum() / total for s in SENTIMENTS])
            valid.append(asp)
        heat_df = pd.DataFrame(matrix, index=valid, columns=SENTIMENTS)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="RdYlGn",
                    linewidths=0.5, ax=ax, vmin=0, vmax=1)
        ax.set_title("Tỷ lệ Sentiment trong từng Entity", fontsize=12, fontweight="bold")
        fig.tight_layout()
        if save:
            fig.savefig(self.out / "imbalance_heatmap.png")
        plt.show()

    # ------------------------------------------------------------------ #
    def missing_label_analysis(self) -> pd.DataFrame:
        present_cols = [f"present_{asp.lower()}" for asp in ASPECTS
                        if f"present_{asp.lower()}" in self.df.columns]
        self.df["n_aspects_labeled"] = self.df[present_cols].sum(axis=1)
        missing_df = self.df[self.df["n_aspects_labeled"] == 0][["review_id", "text", "split"]]
        print(f"\n⚠️  Reviews không có nhãn nào: {len(missing_df)}")
        dist = self.df["n_aspects_labeled"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(dist.index, dist.values, color="#5C6BC0", edgecolor="white")
        ax.set_xlabel("Số entities được gán nhãn"); ax.set_ylabel("Số reviews")
        ax.set_title("Phân phối số entities trên mỗi review", fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        fig.tight_layout()
        fig.savefig(self.out / "missing_labels_dist.png")
        plt.show()
        return missing_df

    # ------------------------------------------------------------------ #
    def plot_aspect_co_occurrence(self, save: bool = True) -> None:
        present_cols = [f"present_{asp.lower()}" for asp in ASPECTS
                        if f"present_{asp.lower()}" in self.df.columns]
        valid = [asp for asp in ASPECTS if f"present_{asp.lower()}" in self.df.columns]
        if not valid:
            print("⚠️ Không tìm thấy cột present_* trong DataFrame")
            return
        co = self.df[present_cols].T.dot(self.df[present_cols])
        co.index = co.columns = valid
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(co, mask=np.eye(len(valid), dtype=bool),
                    annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax)
        ax.set_title("Ma trận đồng xuất hiện Entity", fontweight="bold")
        fig.tight_layout()
        if save:
            fig.savefig(self.out / "aspect_co_occurrence.png")
        plt.show()

    # ------------------------------------------------------------------ #
    def wordcloud_per_sentiment(self, aspect: str = "FOOD", save: bool = True) -> None:
        if not _WC_OK:
            print("⚠️  pip install wordcloud"); return
        col = f"label_{aspect.lower()}"
        if col not in self.df.columns:
            print(f"⚠️  Không tìm thấy cột {col}"); return
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, sent in zip(axes, SENTIMENTS):
            corpus = " ".join(self.df[self.df[col] == sent]["text"].dropna())
            if not corpus.strip():
                ax.text(0.5, 0.5, "Không có dữ liệu", ha="center", va="center",
                        transform=ax.transAxes)
            else:
                wc = WordCloud(width=500, height=300, background_color="white",
                               collocations=False).generate(corpus)
                ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(sent, color=_PALETTE[sent], fontweight="bold")
        fig.suptitle(f"Word Cloud — Entity: {aspect}", fontweight="bold")
        fig.tight_layout()
        if save:
            fig.savefig(self.out / f"wordcloud_{aspect.lower()}.png")
        plt.show()

    # ------------------------------------------------------------------ #
    def run_full_eda(self) -> None:
        print("=" * 60 + "\n  STEP 1.1 — EDA VLSP 2018\n" + "=" * 60)
        self.summary()
        self.plot_text_length()
        self.plot_aspect_distribution()
        self.plot_category_distribution()
        self.class_imbalance_report()
        self.plot_imbalance_heatmap()
        self.missing_label_analysis()
        self.plot_aspect_co_occurrence()
        for asp in ASPECTS[:3]:
            self.wordcloud_per_sentiment(aspect=asp)
        print(f"\n✅ Biểu đồ đã lưu tại: {self.out.resolve()}")