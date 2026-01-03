import os
import glob
import json
from datetime import date
from typing import Dict, Any, List

from engine_backend import build_course_bundle, save_json


KNOWLEDGE_DIR = "knowledge"
STYLE_DIR = "style"

OUT_FULL = "course_materials_full.json"
OUT_BUNDLE = "course_bundle.json"


def read_txt_files(folder: str) -> List[Dict[str, str]]:
    docs = []
    if not os.path.exists(folder):
        return docs

    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            docs.append({"name": os.path.basename(path), "text": f.read()})
    return docs


def main():
    course_id = os.getenv("COURSE_ID") or "course_" + date.today().isoformat()

    knowledge_docs = read_txt_files(KNOWLEDGE_DIR)
    style_docs = read_txt_files(STYLE_DIR)

    if not knowledge_docs:
        raise RuntimeError(f"No .txt found in {KNOWLEDGE_DIR}/")
    if not style_docs:
        raise RuntimeError(f"No .txt found in {STYLE_DIR}/")

    # 1) Build bundle via model (course_profile + terminology + solutions)
    bundle = build_course_bundle(
        course_id=course_id,
        knowledge_docs=knowledge_docs,
        style_docs=style_docs,
        keep_full_texts_in_profile=True,
    )

    # 2) Build FULL materials JSON (raw texts + registry + meta) — independent
    profile = bundle["adaptive_learning_engine_bundle"]["instances"]["active_course_profile"]
    full_materials: Dict[str, Any] = {
        "meta": {
            "course_id": course_id,
            "generated_at": profile.get("meta", {}).get("generated_at"),
            "note": "Full raw materials (converted texts) for the course. Contains ALL docs in full.",
        },
        "doc_registry": profile.get("doc_registry", []),
        "doc_text_by_id": (profile.get("raw_materials", {}) or {}).get("doc_text_by_id", {}),
    }

    save_json(OUT_FULL, full_materials, gzip_if_big=True, big_threshold_mb=10)
    save_json(OUT_BUNDLE, bundle, gzip_if_big=True, big_threshold_mb=10)

    print("✅ Done")
    print(f"- Full materials: {OUT_FULL} (and maybe .gz)")
    print(f"- Operational bundle: {OUT_BUNDLE} (and maybe .gz)")


if __name__ == "__main__":
    main()
