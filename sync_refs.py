"""
sync_refs.py -- extract citations from .tex files and sync them from a .bib collection.

Usage:
    python sync_refs.py --tex-dir <dir> --collection <file.bib> --output <references.bib>

    --tex         Directory to search recursively for .tex files (default: .)
    --collection  Path to your full reference collection .bib file
    --output      Path to the output/existing project .bib file (default: references.bib)
"""

import argparse
import re
import sys
from pathlib import Path

# Citation commands whose argument is a key list (extend if you use others)
CITE_PATTERN = re.compile(
    r"\\(?:cite[a-zA-Z*]*|textcite[a-zA-Z*]*|autocite[a-zA-Z*]*"
    r"|parencite[a-zA-Z*]*|footcite[a-zA-Z*]*)"
    r"(?:\[[^\]]*\])*"   # optional pre/postnotes like \cite[p.~1]{key}
    r"\{([^}]+)\}"
)


def extract_keys_from_tex(tex_dir: Path) -> set[str]:
    keys = set()
    for tex_file in tex_dir.rglob("*.tex"):
        text = tex_file.read_text(encoding="utf-8", errors="replace")
        for match in CITE_PATTERN.finditer(text):
            for key in match.group(1).split(","):
                keys.add(key.strip())
    return keys


def parse_bib(path: Path) -> dict[str, str]:
    """Return dict mapping BibTeX key -> full entry string."""
    entries = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    i = 0
    n = len(text)
    while i < n:
        # Find start of an entry
        m = re.search(r"@[A-Za-z]+[ \t]*\{", text[i:])
        if not m:
            break
        start = i + m.start()
        # Count braces to find the matching closing brace
        depth = 0
        for j in range(start, n):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    entry = text[start : j + 1]
                    key_match = re.match(r"@[A-Za-z]+[ \t]*\{([^,]+),", entry)
                    if key_match:
                        key = key_match.group(1).strip()
                        entries[key] = entry
                    i = j + 1
                    break
        else:
            # Unmatched brace -- skip past this @
            i = start + m.end()
    return entries


def main():
    parser = argparse.ArgumentParser(description="Sync BibTeX entries from a collection into a project .bib file.")
    parser.add_argument("--tex", type=Path, default=Path("."), help="Directory to search for .tex files")
    parser.add_argument("--collection", type=Path, required=True, help="Full reference collection .bib file")
    parser.add_argument("--output", type=Path, default=Path("references.bib"), help="Project .bib file to create or update")
    args = parser.parse_args()

    # 1. Extract citation keys from .tex files
    print(f"Scanning .tex files in: {args.tex.resolve()}")
    needed_keys = extract_keys_from_tex(args.tex)
    if not needed_keys:
        print("No citation keys found in .tex files.")
        sys.exit(0)
    print(f"Found {len(needed_keys)} unique citation key(s).")

    # 2. Parse the collection
    print(f"Parsing collection: {args.collection}")
    collection = parse_bib(args.collection)
    print(f"Collection contains {len(collection)} entries.")

    # 3. Load existing output .bib (if present) to avoid duplicates
    existing = {}
    if args.output.exists():
        print(f"Loading existing output: {args.output}")
        existing = parse_bib(args.output)
        print(f"Existing output contains {len(existing)} entries.")

    # 4. Determine what to add
    missing_from_collection = []
    already_present = []
    to_add = {}

    for key in sorted(needed_keys):
        if key in existing:
            already_present.append(key)
        elif key in collection:
            to_add[key] = collection[key]
        else:
            missing_from_collection.append(key)

    print(f"\n{len(already_present)} key(s) already in output .bib -- skipped.")
    print(f"{len(to_add)} key(s) to add from collection.")
    if missing_from_collection:
        print(f"\nWARNING: {len(missing_from_collection)} key(s) not found in collection:")
        for k in missing_from_collection:
            print(f"  {k}")

    if not to_add:
        print("\nNothing to add. Output file unchanged.")
        return

    # 5. Merge and sort alphabetically by key
    merged = {**existing, **to_add}
    sorted_entries = [merged[k] for k in sorted(merged, key=str.casefold)]
    output_text = "\n\n".join(sorted_entries) + "\n"

    args.output.write_text(output_text, encoding="utf-8")
    print(f"\nWrote {len(merged)} entries to {args.output} (added {len(to_add)}).")


if __name__ == "__main__":
    main()
