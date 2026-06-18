# src/utils/jsonl.py
import json


def load_jsonl(file_path: str) -> list[dict]:
    samples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    return samples


def save_jsonl(file_path: str, rows: list[dict]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")