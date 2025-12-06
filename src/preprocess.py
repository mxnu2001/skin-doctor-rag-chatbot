# src/preprocess.py
import pandas as pd
import json
from pathlib import Path
from typing import List

def clean_text(text: str) -> str:
    if text is None:
        return ""
    # Minimal cleaning; tune as needed
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())

def split_text(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks

def process_csv(
    csv_path: str,
    text_column: str = "Doctor_Style_Answer",
    output_dir: str = "data/processed",
    chunk_size: int = 250,
    overlap: int = 50,
):
    """
    Read CSV at `csv_path`, extract `text_column`, split into chunks,
    and write chunk JSON files into output_dir.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str).fillna("")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created_files = []
    for idx, row in df.iterrows():
        raw = row.get(text_column, "")
        text = clean_text(raw)
        if not text:
            continue
        chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
        for c_i, chunk in enumerate(chunks):
            obj = {"text": chunk, "meta": {"row_id": int(idx), "chunk_id": c_i}}
            fname = f"row{idx}_chunk{c_i}.json"
            fp = out_dir / fname
            fp.write_text(json.dumps(obj, ensure_ascii=False))
            created_files.append(str(fp))
    return created_files

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/derma_data.csv", help="Path to CSV file")
    parser.add_argument("--col", default="Doctor_Style_Answer", help="Column to use for text")
    parser.add_argument("--out", default="data/processed", help="Output folder for chunk JSONs")
    parser.add_argument("--chunk_size", type=int, default=250)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    created = process_csv(args.csv, text_column=args.col, output_dir=args.out,
                          chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Created {len(created)} chunk files in {args.out}")
