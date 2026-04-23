"""
VLSP 2018 Preprocessor — Step 1.2
Đóng gói toàn bộ pipeline tiền xử lý:
  1. Load CSV/TXT (VLSP 2018 format)
  2. Unicode normalization (NFC)
  3. Text cleaning (URL, emoji, dup whitespace)
  4. Vietnamese tokenization (underthesea / fallback pyvi / split)
  5. Chuẩn hóa labels positive/negative/neutral
  6. Xử lý class imbalance (class weights / oversampling)
  7. Train/val/test split với stratify
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALID_LABELS = {"positive", "negative", "neutral"}
SENT2INT = {"positive": 1, "neutral": 0, "negative": -1, "no_mention": -2}

# ---------- Vietnamese tokenizer (fallback chain) ----------
_TOKENIZER = None
_TOKENIZER_NAME = None

def _load_tokenizer():
    """Lazy-load Vietnamese tokenizer. Né lỗi OSError của torch khi underthesea hỏng."""
    disable_uts = os.environ.get('DISABLE_UNDERTHESEA', '').strip() in ('1', 'true', 'True')

    if not disable_uts:
        try:
            from underthesea import word_tokenize as _uts_tokenize
            return 'underthesea', lambda s: _uts_tokenize(s, format='text')
        except (ImportError, OSError) as e:
            # OSError: torch DLL init failed trên Windows
            print(f'[preprocessor] underthesea không khả dụng ({e.__class__.__name__}); thử pyvi...')

    try:
        from pyvi import ViTokenizer
        return 'pyvi', lambda s: ViTokenizer.tokenize(s)
    except ImportError as e:
        raise RuntimeError(
            "Không có tokenizer tiếng Việt. Cài: pip install pyvi"
        ) from e

def _get_tokenizer():
    """Lazy init tokenizer: underthesea → pyvi → whitespace."""
    global _TOKENIZER, _TOKENIZER_NAME
    if _TOKENIZER is not None:
        return _TOKENIZER, _TOKENIZER_NAME

    _TOKENIZER_NAME, _TOKENIZER = _load_tokenizer()

    return _TOKENIZER, _TOKENIZER_NAME


# ---------- Regex patterns ----------
_URL_RE   = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_EMOJI_RE = re.compile(
    "[" "\U0001F600-\U0001F64F"   # emoticons
        "\U0001F300-\U0001F5FF"   # symbols
        "\U0001F680-\U0001F6FF"   # transport
        "\U0001F1E0-\U0001F1FF"   # flags
        "\U00002500-\U00002BEF"
        "\U00002702-\U000027B0"
        "\U0001F900-\U0001F9FF"
    "]+", flags=re.UNICODE,
)
_MULTI_WS_RE = re.compile(r"\s+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")  # "ngonnnn" → "ngonn"


# ============================================================ #
#  Config                                                        #
# ============================================================ #

@dataclass
class PreprocessConfig:
    lowercase: bool        = True
    nfc_normalize: bool    = True
    remove_url: bool       = True
    remove_email: bool     = True
    remove_emoji: bool     = False   # giữ emoji có thể hữu ích cho sentiment
    collapse_repeat: bool  = True
    tokenize: bool         = True
    min_words: int         = 2       # bỏ review quá ngắn
    max_words: int         = 256
    random_state: int      = 42


# ============================================================ #
#  Main Preprocessor                                             #
# ============================================================ #

class VLSPPreprocessor:
    """Pipeline tiền xử lý cho VLSP 2018 Restaurant SA."""

    def __init__(self, config: PreprocessConfig | None = None):
        self.cfg = config or PreprocessConfig()
        self.review_col: str | None = None
        self.aspect_cols: list[str] = []
        self.format_type: str | None = None    # 'dense' | 'sparse'
        self.tokenizer_name: str | None = None

    # ---------------- Loaders ---------------- #

    @staticmethod
    def _load_txt_vlsp(path: Path) -> pd.DataFrame:
        """Parse VLSP 2018 TXT: 3-line blocks với {ENTITY#ATTR, polarity}."""
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            blocks = fh.read().strip().split("\n\n")
        for blk in blocks:
            lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
            if len(lines) < 2:
                continue
            if lines[0].startswith("#"):
                rid = lines[0].lstrip("#").strip()
                text = lines[1]
                label_line = lines[2] if len(lines) >= 3 else ""
            else:
                rid = str(len(records))
                text = lines[0]
                label_line = lines[1] if len(lines) >= 2 else ""
            row = {"review_id": rid, "text": text}
            for m in re.finditer(r"\{([^,}]+),\s*([^}]+)\}", label_line):
                cat = m.group(1).strip().upper()
                pol = m.group(2).strip().lower()
                row[cat] = pol
            records.append(row)
        return pd.DataFrame(records)

    def load(self, path: str | Path) -> pd.DataFrame:
        """Load file CSV hoặc TXT (VLSP format)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        else:
            df = self._load_txt_vlsp(p)
        logger.info(f"Loaded {len(df):,} rows from {p.name}")
        return df

    # ---------------- Schema detection ---------------- #

    def detect_schema(self, df: pd.DataFrame) -> tuple[str, list[str]]:
        """Auto-detect review_col (text) và aspect_cols (labels)."""
        stats = {}
        for col in df.columns:
            s = df[col].astype(str)
            stats[col] = (s.str.len().mean(), s.nunique() / max(len(s), 1))

        candidates = {c: v for c, v in stats.items() if v[1] > 0.5}
        review_col = (
            max(candidates, key=lambda c: candidates[c][0])
            if candidates
            else max(stats, key=lambda c: stats[c][0])
        )
        aspect_cols = [c for c in df.columns if c != review_col]
        self.review_col  = review_col
        self.aspect_cols = aspect_cols
        return review_col, aspect_cols

    # ---------------- Text cleaning ---------------- #

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        if self.cfg.nfc_normalize:
            text = unicodedata.normalize("NFC", text)
        if self.cfg.remove_url:
            text = _URL_RE.sub(" ", text)
        if self.cfg.remove_email:
            text = _EMAIL_RE.sub(" ", text)
        if self.cfg.remove_emoji:
            text = _EMOJI_RE.sub(" ", text)
        if self.cfg.collapse_repeat:
            text = _REPEAT_CHAR_RE.sub(r"\1\1", text)
        if self.cfg.lowercase:
            text = text.lower()
        text = _MULTI_WS_RE.sub(" ", text).strip()
        return text

    def _tokenize(self, text: str) -> str:
        if not self.cfg.tokenize or not text:
            return text
        tok, name = _get_tokenizer()
        self.tokenizer_name = name
        try:
            return tok(text)
        except Exception as e:
            logger.warning(f"Tokenize failed: {e}")
            return text

    # ---------------- Label normalization ---------------- #

    def _normalize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.aspect_cols:
            cleaned = df[col].astype(str).str.strip().str.lower()
            df[col] = cleaned.where(cleaned.isin(VALID_LABELS), other=np.nan)
        # Loại aspect không có nhãn hợp lệ
        kept = [c for c in self.aspect_cols if df[c].notna().sum() > 0]
        self.aspect_cols = kept

        # Detect format type
        ratios = {c: df[c].notna().mean() for c in self.aspect_cols}
        self.format_type = "dense" if all(r > 0.99 for r in ratios.values()) else "sparse"
        return df

    # ---------------- Full pipeline ---------------- #

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline đầy đủ:
          detect schema → normalize labels → clean text → tokenize → filter
        """
        if self.review_col is None:
            self.detect_schema(df)

        df = df.copy()
        df = self._normalize_labels(df)

        # Clean text
        df["text_clean"]     = df[self.review_col].astype(str).map(self._clean_text)
        df["text_tokenized"] = df["text_clean"].map(self._tokenize)
        df["word_count"]     = df["text_tokenized"].str.split().str.len()
        df["char_count"]     = df["text_clean"].str.len()

        # Filter
        before = len(df)
        df = df[
            (df["word_count"] >= self.cfg.min_words)
            & (df["word_count"] <= self.cfg.max_words)
        ].reset_index(drop=True)
        logger.info(f"Filter by length: {before} → {len(df)} rows")

        return df

    # ---------------- Class weights ---------------- #

    def compute_class_weights(self, df: pd.DataFrame) -> dict[str, dict]:
        """
        Tính class_weight cho mỗi aspect theo công thức sklearn balanced:
            w_c = n_total / (n_classes * n_c)
        """
        weights = {}
        for asp in self.aspect_cols:
            vc = df[asp].value_counts()
            n  = vc.sum()
            k  = len(vc)
            if n == 0 or k == 0:
                continue
            weights[asp] = {cls: round(n / (k * cnt), 4) for cls, cnt in vc.items()}
        return weights

    # ---------------- Stratified split ---------------- #

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size:  float = 0.15,
        stratify_on: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Chia train/val/test với stratify theo một aspect (nếu chỉ định).
        Nếu không, chia random theo random_state.
        """
        from sklearn.model_selection import train_test_split

        strat = df[stratify_on] if stratify_on and stratify_on in df.columns else None
        df_tv, df_test = train_test_split(
            df, test_size=test_size, random_state=self.cfg.random_state, stratify=strat
        )

        strat_tv = df_tv[stratify_on] if stratify_on and stratify_on in df_tv.columns else None
        val_rel  = val_size / (1 - test_size)
        df_train, df_val = train_test_split(
            df_tv, test_size=val_rel, random_state=self.cfg.random_state, stratify=strat_tv
        )
        return {
            "train": df_train.reset_index(drop=True),
            "val":   df_val.reset_index(drop=True),
            "test":  df_test.reset_index(drop=True),
        }

    # ---------------- Save ---------------- #

    @staticmethod
    def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in splits.items():
            path = out / f"{name}.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"✅ {name}: {len(df):,} rows → {path}")

    # ---------------- Summary ---------------- #

    def summary(self, df: pd.DataFrame) -> dict:
        return {
            "n_rows":         len(df),
            "review_col":     self.review_col,
            "aspect_cols":    self.aspect_cols,
            "format_type":    self.format_type,
            "tokenizer":      self.tokenizer_name,
            "avg_words":      round(df["word_count"].mean(), 2) if "word_count" in df else None,
            "median_words":   int(df["word_count"].median())    if "word_count" in df else None,
            "pct_over_128":   round((df["word_count"] > 128).mean(), 4) if "word_count" in df else None,
        }
