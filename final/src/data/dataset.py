"""
VLSP 2018 Restaurant SA — Data Loader
Hỗ trợ tự động phát hiện format: XML hoặc TXT
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

ASPECT_CATEGORIES = [
    "FOOD#QUALITY", "FOOD#PRICE", "FOOD#STYLE_OPTIONS",
    "DRINKS#QUALITY", "DRINKS#PRICES", "DRINKS#STYLE_OPTIONS",
    "RESTAURANT#PRICES", "RESTAURANT#GENERAL", "RESTAURANT#MISCELLANEOUS",
    "AMBIENCE#GENERAL", "LOCATION#GENERAL", "SERVICE#GENERAL",
]
ENTITIES   = ["FOOD", "DRINKS", "RESTAURANT", "AMBIENCE", "LOCATION", "SERVICE"]
SENTIMENTS = ["positive", "negative", "neutral"]
_SENT2INT  = {"positive": 1, "neutral": 0, "negative": -1}
_PRIORITY  = {"negative": 0, "positive": 1, "neutral": 2}


@dataclass
class ReviewRecord:
    rid: str
    text: str
    category_labels: dict[str, str] = field(default_factory=dict)

    def _entity_label(self, entity: str) -> str | None:
        sents = [v for k, v in self.category_labels.items() if k.startswith(entity + "#")]
        if not sents:
            return None
        return min(sents, key=lambda s: _PRIORITY.get(s, 99))

    def to_flat_dict(self) -> dict:
        row: dict = {"review_id": self.rid, "text": self.text}
        for cat in ASPECT_CATEGORIES:
            col = "cat_" + cat.lower().replace("#", "_")
            row[col] = self.category_labels.get(cat)
        for ent in ENTITIES:
            lbl = self._entity_label(ent)
            row[f"label_{ent.lower()}"]   = lbl
            row[f"score_{ent.lower()}"]   = _SENT2INT.get(lbl, 0)
            row[f"present_{ent.lower()}"] = int(lbl is not None)
        return row


class VLSP2018Dataset:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self._vcfg = cfg["vlsp2018"]

    # ------------------------------------------------------------------ #
    # Inspect — gọi từ notebook để xem format thực tế                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def inspect_file(filepath: str, n_lines: int = 10) -> None:
        """In ra n dòng đầu và tự đoán format."""
        p = Path(filepath)
        if not p.exists():
            print(f"❌ Không tìm thấy: {filepath}")
            return

        print(f"\n{'='*60}")
        print(f"📄 {p.name}  ({p.stat().st_size/1024:.1f} KB)")
        print(f"{'='*60}")

        # Thử UTF-8 trước, fallback utf-8-sig (BOM), rồi latin-1
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                with open(p, "r", encoding=enc) as fh:
                    lines = [fh.readline() for _ in range(n_lines)]
                print(f"Encoding : {enc}")
                break
            except UnicodeDecodeError:
                continue

        # Phát hiện format
        first = lines[0].strip() if lines else ""
        if first.startswith("<"):
            fmt = "XML"
        elif "\t" in first:
            fmt = "TXT (tab-separated)"
        else:
            fmt = "TXT (unknown separator)"

        print(f"Format   : {fmt}")
        print(f"\n--- {n_lines} dòng đầu ---")
        for i, ln in enumerate(lines, 1):
            print(f"{i:>3}: {ln}", end="")

        # Thống kê nhanh nếu TXT
        if "TXT" in fmt:
            print(f"\n--- Phân tích cột ---")
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    with open(p, "r", encoding=enc) as fh:
                        sample = [ln.strip() for ln in fh if ln.strip()][:200]
                    break
                except UnicodeDecodeError:
                    continue

            n_cols = [len(ln.split("\t")) for ln in sample]
            print(f"Số cột (tab) thường gặp: {max(set(n_cols), key=n_cols.count)}")

            # Tìm tất cả categories trong file
            cats: set[str] = set()
            for ln in sample:
                parts = ln.split("\t")
                if len(parts) >= 2:
                    for tok in parts[1].split(","):
                        if ":" in tok:
                            cats.add(tok.rsplit(":", 1)[0].strip().upper())
            print(f"Categories tìm thấy  : {sorted(cats)}")

    # ------------------------------------------------------------------ #
    # Parsers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_txt_line(line: str, idx: int) -> ReviewRecord | None:
        """
        Hỗ trợ 2 format TXT của VLSP 2018:
          Format A: {text}\t{CAT:pol,CAT:pol}
          Format B: {text}\t{CAT}\t{polarity}   (single aspect per line)
        """
        line = line.strip()
        if not line:
            return None
        parts = line.split("\t")
        text  = parts[0].strip()
        if not text:
            return None

        labels: dict[str, str] = {}

        if len(parts) == 3:
            # Format B: text \t CATEGORY \t polarity
            cat  = parts[1].strip().upper()
            sent = parts[2].strip().lower()
            if sent in SENTIMENTS:
                labels[cat] = sent

        elif len(parts) >= 2 and parts[1].strip():
            # Format A: text \t CAT:pol,CAT:pol,...
            for token in parts[1].split(","):
                token = token.strip()
                if ":" not in token:
                    continue
                cat, sent = token.rsplit(":", 1)
                cat  = cat.strip().upper()
                sent = sent.strip().lower()
                if sent in SENTIMENTS:
                    if cat not in labels or _PRIORITY.get(sent, 99) < _PRIORITY.get(labels[cat], 99):
                        labels[cat] = sent

        return ReviewRecord(rid=str(idx), text=text, category_labels=labels)

    @staticmethod
    def _parse_xml_file(filepath: str) -> list[ReviewRecord]:
        """Parse VLSP XML format."""
        records = []
        root = ET.parse(filepath).getroot()
        for i, rev in enumerate(root.findall(".//Review")):
            rid  = rev.get("rid", str(i))
            text_node = rev.find("text")
            text = (text_node.text or "").strip() if text_node is not None else ""
            labels: dict[str, str] = {}
            for op in rev.findall(".//Opinion"):
                cat  = op.get("category", "").upper()
                sent = op.get("polarity", "").lower()
                if sent in SENTIMENTS:
                    if cat not in labels or _PRIORITY.get(sent, 99) < _PRIORITY.get(labels[cat], 99):
                        labels[cat] = sent
            records.append(ReviewRecord(rid=rid, text=text, category_labels=labels))
        return records

    def _read(self, filepath: str) -> list[ReviewRecord]:
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy: {filepath}")

        # Phát hiện format từ nội dung thực tế
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                with open(p, "r", encoding=enc) as fh:
                    first_line = fh.readline()
                self._encoding = enc
                break
            except UnicodeDecodeError:
                continue

        if first_line.strip().startswith("<"):
            return self._parse_xml_file(filepath)

        # TXT format
        records = []
        with open(p, "r", encoding=self._encoding) as fh:
            for idx, line in enumerate(fh):
                rec = self._parse_txt_line(line, idx)
                if rec is not None:
                    records.append(rec)
        return records

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(self, split: str = "train") -> pd.DataFrame:
        fpath = self._vcfg.get(f"{split}_file")
        if not fpath:
            raise ValueError(f"Không có config key: {split}_file")
        rows = [r.to_flat_dict() for r in self._read(fpath)]
        if not rows:
            raise ValueError(f"File rỗng hoặc không parse được: {fpath}")
        df = pd.DataFrame(rows)
        df["text_len"]   = df["text"].str.len()
        df["word_count"] = df["text"].str.split().str.len()
        df["split"]      = split
        return df

    def load_all(self) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for sp in ("train", "dev", "test"):
            try:
                out[sp] = self.load(sp)
                print(f"✅ [{sp:5s}] {len(out[sp]):>5,} reviews")
            except (FileNotFoundError, ValueError) as e:
                print(f"⚠️  {e}")
        return out


VLSP2018Loader = VLSP2018Dataset
