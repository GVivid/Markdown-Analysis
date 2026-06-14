import os
import hashlib
import re
import duckdb
import argparse
import shutil
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering


def clean_markdown(text: str) -> str:
    """Strips basic markdown syntax to normalize textual payload."""
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"#+\s+", "", text)
    text = re.sub(r"[*_`~]+", "", text)
    return text.strip().lower()


def compute_md5(text: str) -> str:
    """Generates MD5 hash from normalized content for exact match filtering."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def setup_database(con: duckdb.DuckDBPyConnection):
    """Initializes schema for incremental state tracking."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            filepath TEXT PRIMARY KEY,
            raw_content TEXT,
            cleaned_content TEXT,
            md5_hash TEXT,
            mtime DOUBLE,
            cluster_id INTEGER,
            is_canonical BOOLEAN DEFAULT FALSE
        );
    """)


def sync_target_directory(con: duckdb.DuckDBPyConnection, target_path: Path):
    """Ingests changed files using mtime validation and purges deleted records."""
    # Step A: Purge stale database records for files no longer on disk
    db_paths = [r[0] for r in con.execute("SELECT filepath FROM documents").fetchall()]
    for path_str in db_paths:
        if not Path(path_str).exists():
            con.execute("DELETE FROM documents WHERE filepath = ?", [path_str])
            print(f"Removed missing file from tracking: {path_str}")

    # Step B: Incremental update loop
    for file in target_path.glob("**/*.md"):
        try:
            current_mtime = file.stat().st_mtime

            # Check for existing state matching the current modification time
            row = con.execute(
                "SELECT mtime FROM documents WHERE filepath = ?", [str(file)]
            ).fetchone()

            if row and row[0] == current_mtime:
                continue  # File content has not changed; skip parsing

            raw_text = file.read_text(encoding="utf-8")
            cleaned = clean_markdown(raw_text)
            md5_str = compute_md5(cleaned)

            con.execute(
                """
                INSERT INTO documents (filepath, raw_content, cleaned_content, md5_hash, mtime, is_canonical, cluster_id)
                VALUES (?, ?, ?, ?, ?, FALSE, NULL)
                ON CONFLICT (filepath) DO UPDATE SET
                    raw_content = EXCLUDED.raw_content,
                    cleaned_content = EXCLUDED.cleaned_content,
                    md5_hash = EXCLUDED.md5_hash,
                    mtime = EXCLUDED.mtime,
                    is_canonical = FALSE,
                    cluster_id = NULL
            """,
                (str(file), raw_text, cleaned, md5_str, current_mtime),
            )
            print(f"Parsed/Updated track state: {file.name}")
        except Exception as e:
            print(f"Failed processing {file}: {e}")


def execute_physical_file_ops(
    con: duckdb.DuckDBPyConnection, action: str, target_dir: Path
):
    """Applies the consolidation strategy to disk storage architecture."""
    if action == "report":
        return

    drop_files = [
        r[0]
        for r in con.execute(
            "SELECT filepath FROM documents WHERE is_canonical = FALSE AND cluster_id IS NOT NULL"
        ).fetchall()
    ]

    if not drop_files:
        print("No redundant file paths targeted for consolidation operations.")
        return

    if action == "archive":
        archive_root = target_dir / ".dedup_archive"
        archive_root.mkdir(exist_ok=True)
        print(f"Orchestrating archival routines. Destination: {archive_root}/")

        for file_str in drop_files:
            src_path = Path(file_str)
            if src_path.exists():
                # Prevent path collisions inside archive pool using hashed names
                unique_name = (
                    f"{hashlib.md5(file_str.encode()).hexdigest()[:8]}_{src_path.name}"
                )
                shutil.move(src_path, archive_root / unique_name)

    elif action == "delete":
        print(
            f"Executing hard erasure routine across {len(drop_files)} target points..."
        )
        for file_str in drop_files:
            src_path = Path(file_str)
            if src_path.exists():
                src_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Incremental Multi-Stage Markdown Semantic Deduplication Tool."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./my_files",
        help="Target processing directory path.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="markdown_analysis.db",
        help="DuckDB state cache database path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Semantic proximity cosine merge cap (0.0 - 1.0).",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["report", "archive", "delete"],
        default="report",
        help="Execution strategy: report only, move to archive directory, or permanent deletion.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Drops local database tracking state before running.",
    )
    args = parser.parse_args()

    target_path = Path(args.dir)
    if not target_path.exists():
        print(f"Error: Target directory path '{args.dir}' does not exist.")
        return

    if args.clear_cache and Path(args.db).exists():
        Path(args.db).unlink()
        print("Cleared database state cache successfully.")

    con = duckdb.connect(database=args.db)
    setup_database(con)

    print(f"Syncing analytical store with: {target_path.resolve()}")
    sync_target_directory(con, target_path)

    # Stage 1: Exact Duplicate Processing
    print("Executing Stage 1: MD5 exact match consolidation...")
    con.execute("UPDATE documents SET is_canonical = FALSE")  # Reset validation tree
    hashes = con.execute("SELECT DISTINCT md5_hash FROM documents").fetchall()

    for (md5_val,) in hashes:
        matching_files = con.execute(
            "SELECT filepath FROM documents WHERE md5_hash = ? ORDER BY filepath ASC",
            [md5_val],
        ).fetchall()
        canonical_file = matching_files[0][0]
        con.execute(
            "UPDATE documents SET is_canonical = TRUE WHERE filepath = ?",
            [canonical_file],
        )

    # Stage 2: Conceptual Overlap via LSA
    unique_docs = con.execute(
        "SELECT filepath, cleaned_content FROM documents WHERE is_canonical = TRUE"
    ).fetchall()

    if len(unique_docs) < 2:
        print("Insufficient unique documents to run semantic clustering steps.")
        con.close()
        return

    filepaths, contents = zip(*unique_docs)

    print(
        "Executing Stage 2: Transforming text via TF-IDF + Latent Semantic Analysis..."
    )
    vectorizer = TfidfVectorizer(
        stop_words="english", sublinear_tf=True, max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(contents)

    n_components = min(100, tfidf_matrix.shape[0] - 1)
    if n_components > 1:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        conceptual_matrix = svd.fit_transform(tfidf_matrix)
    else:
        conceptual_matrix = tfidf_matrix.toarray()

    norms = np.linalg.norm(conceptual_matrix, axis=1)
    zero_indices = np.where(norms == 0)[0]
    valid_indices = np.where(norms > 0)[0]

    labels = np.full(len(filepaths), -1, dtype=int)
    max_label = 0

    if len(valid_indices) >= 2:
        print(
            "Computing matrix distances and fitting agglomerative tree cluster layer..."
        )
        valid_matrix = conceptual_matrix[valid_indices]
        distance_threshold = 1.0 - args.threshold

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels[valid_indices] = clustering.fit_predict(valid_matrix)
        max_label = int(np.max(labels[valid_indices]))
    elif len(valid_indices) == 1:
        labels[valid_indices] = 0
        max_label = 0

    if len(zero_indices) > 0:
        empty_cluster_id = max_label + 1
        labels[zero_indices] = empty_cluster_id
        print(
            f"Isolated {len(zero_indices)} empty/stopword-only files into specialized group (Cluster #{empty_cluster_id})."
        )

    for filepath, label in zip(filepaths, labels):
        con.execute(
            "UPDATE documents SET cluster_id = ? WHERE filepath = ?",
            (int(label), filepath),
        )

    print("Resolving final consolidation mappings...")
    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]

        if len(cluster_indices) > 1:
            cluster_vectors = conceptual_matrix[cluster_indices]
            centroid = np.mean(cluster_vectors, axis=0)

            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm == 0:
                best_idx = cluster_indices[0]
            else:
                similarities = np.dot(cluster_vectors, centroid) / (
                    np.linalg.norm(cluster_vectors, axis=1) * centroid_norm
                )
                best_idx = cluster_indices[np.argmax(similarities)]

            chosen_canonical = filepaths[best_idx]
            cluster_paths = [filepaths[i] for i in cluster_indices]
            con.execute(
                "UPDATE documents SET is_canonical = FALSE WHERE filepath IN (SELECT UNNEST(?))",
                [cluster_paths],
            )
            con.execute(
                "UPDATE documents SET is_canonical = TRUE WHERE filepath = ?",
                [chosen_canonical],
            )

    # Run Database Update to catch Stage 1 files wrapped by exact matches
    con.execute("""
        UPDATE documents 
        SET cluster_id = (SELECT cluster_id FROM documents d2 WHERE d2.md5_hash = documents.md5_hash AND d2.cluster_id IS NOT NULL LIMIT 1)
        WHERE cluster_id IS NULL
    """)

    print("\n" + "=" * 50 + "\nPIPELINE REDUCTION SUMMARY REPORT\n" + "=" * 50)
    report_query = """
        SELECT 
            cluster_id,
            MAX(CASE WHEN is_canonical THEN filepath END) as retain_file,
            string_agg(CASE WHEN NOT is_canonical THEN filepath END, ', ') as duplicate_files,
            COUNT(*) as total_group_count
        FROM documents 
        GROUP BY cluster_id
        HAVING total_group_count > 1
    """
    results = con.execute(report_query).fetchall()

    if not results:
        print(
            "No duplicate configurations or conceptual clusters exceeded threshold parameters."
        )
    else:
        for cid, retain, dupes, count in results:
            print(f"\n[Conceptual Cluster #{cid}] -> Group Count: {count} files")
            print(f"  [KEEP] --> {retain}")
            print(f"  [DROP] --> {dupes}")

    # Step 3: Trigger Physical Disk Transformations
    execute_physical_file_ops(con, args.action, target_path)
    con.close()


if __name__ == "__main__":
    main()
