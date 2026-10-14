"""
Rebuild the 75-problem LiveCodeBench cache used by run_lcb_canonical.py.

Loads livecodebench/code_generation_lite (release from the manifest, release_v6)
from Hugging Face, keeps the manifest's ordered QIDs, and writes the same record
schema as the pinned cache. The rebuilt file is checked against the manifest's
problem_cache_content_sha256; the byte hash is not expected to match.

Requires network access and: datasets==3.6.0 plus the lcb_runner dependencies
(pebble, tqdm). These are not in requirements.txt.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "LiveCodeBench"))

# Must match CONTENT_FIELDS and cache_content_sha256 in run_lcb_canonical.py.
CONTENT_FIELDS = (
    "question_id",
    "question_title",
    "question_content",
    "difficulty",
    "platform",
    "starter_code",
    "metadata",
    "public_test_cases",
    "private_test_cases",
)


def content_sha256(records):
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "benchmarks" / "LiveCodeBench" / "data" / "lcb75_prior_manifest.json"),
    )
    parser.add_argument("--out", default=None, help="Output path (default: release root / manifest problem_cache)")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else ROOT / manifest["problem_cache"]
    qids = manifest["ordered_question_ids"]

    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

    problems = load_code_generation_dataset(manifest["release"])
    qid_set = set(qids)
    wanted = {p.question_id: p for p in problems if p.question_id in qid_set}
    missing = [q for q in qids if q not in wanted]
    if missing:
        raise SystemExit(f"QIDs not found in {manifest['release']}: {missing[:5]}")

    records = []
    for qid in qids:
        p = wanted[qid]
        records.append({
            "question_id": p.question_id,
            "question_title": p.question_title,
            "question_content": p.question_content,
            "difficulty": p.difficulty.value,
            "platform": p.platform.value,
            "starter_code": p.starter_code,
            "metadata": p.metadata,
            "public_test_cases": [{"input": t.input, "output": t.output} for t in p.public_test_cases],
            "private_test_cases": [{"input": t.input, "output": t.output} for t in p.private_test_cases],
        })

    out.write_text(
        json.dumps({"_meta": {"source": "livecodebench/code_generation_lite",
                              "release": manifest["release"],
                              "count": len(records)},
                    "problems": records}, ensure_ascii=False),
        encoding="utf-8",
    )

    got = content_sha256([{f: r.get(f) for f in CONTENT_FIELDS} for r in records])
    expected = manifest.get("problem_cache_content_sha256")
    print(f"wrote {out} ({len(records)} problems)")
    print(f"content_sha256 {got}")
    print(f"manifest       {expected}")
    print("byte sha256    " + hashlib.sha256(out.read_bytes()).hexdigest() + " (informational)")
    if got != expected:
        raise SystemExit("MISMATCH: rebuilt records do not match the manifest content pin")
    print("OK: rebuilt records match the manifest content pin")


if __name__ == "__main__":
    main()
