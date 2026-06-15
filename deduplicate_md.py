# Import standard libraries for OS interaction, hashing, and regex processing
import argparse
import hashlib
import shutil
from pathlib import Path

# Import scientific computing and machine learning libraries
import duckdb
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class SQLQueries:
    """Centralized repository for database schema definitions and SQL statements."""

    CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS documents (
            filepath TEXT PRIMARY KEY,
            raw_content TEXT,
            cleaned_content TEXT,
            md5_hash TEXT,
            mtime DOUBLE,
            cluster_id INTEGER,
            is_canonical BOOLEAN DEFAULT FALSE
        );
    """

    SELECT_ALL_FILEPATHS = "SELECT filepath FROM documents"
    DELETE_BY_FILEPATH = "DELETE FROM documents WHERE filepath = ?"
    SELECT_MTIME_BY_FILEPATH = "SELECT mtime FROM documents WHERE filepath = ?"

    UPSERT_DOCUMENT = """
        INSERT INTO documents (filepath, raw_content, cleaned_content, md5_hash, mtime, is_canonical, cluster_id)
        VALUES (?, ?, ?, ?, ?, FALSE, NULL)
        ON CONFLICT (filepath) DO UPDATE SET
            raw_content = EXCLUDED.raw_content,
            cleaned_content = EXCLUDED.cleaned_content,
            md5_hash = EXCLUDED.md5_hash,
            mtime = EXCLUDED.mtime,
            is_canonical = FALSE,
            cluster_id = NULL
    """

    RESET_CANONICAL_FLAGS = "UPDATE documents SET is_canonical = FALSE"
    SELECT_DISTINCT_MD5 = "SELECT DISTINCT md5_hash FROM documents"
    SELECT_FILEPATHS_BY_MD5 = (
        "SELECT filepath FROM documents WHERE md5_hash = ? ORDER BY filepath ASC"
    )
    SET_CANONICAL_TRUE = "UPDATE documents SET is_canonical = TRUE WHERE filepath = ?"

    UPDATE_CLUSTER_ID = "UPDATE documents SET cluster_id = ? WHERE filepath = ?"
    SET_CLUSTER_CANONICAL_FALSE = (
        "UPDATE documents SET is_canonical = FALSE WHERE filepath IN (SELECT UNNEST(?))"
    )

    PROPAGATE_EXACT_DUPLICATE_CLUSTERS = """
        UPDATE documents 
        SET cluster_id = (
            SELECT cluster_id 
            FROM documents d2 
            WHERE d2.md5_hash = documents.md5_hash 
              AND d2.cluster_id IS NOT NULL 
            LIMIT 1
        )
        WHERE cluster_id IS NULL
    """

    SELECT_NON_CANONICAL_CLUSTERED = """
        SELECT filepath 
        FROM documents 
        WHERE is_canonical = FALSE 
          AND cluster_id IS NOT NULL
    """

    PIPELINE_REDUCTION_REPORT = """
        SELECT 
            cluster_id,
            MAX(CASE WHEN is_canonical THEN filepath END) as retain_file,
            string_agg(CASE WHEN NOT is_canonical THEN filepath END, ', ') as duplicate_files,
            COUNT(*) as total_group_count
        FROM documents 
        GROUP BY cluster_id
        HAVING total_group_count > 1
    """

    SELECT_CANONICAL_DOCUMENTS = (
        "SELECT filepath, cleaned_content FROM documents WHERE is_canonical = TRUE"
    )


def clean_markdown(text: str) -> str:
    """Strips basic markdown syntax using a sequential character scanner."""
    # Step 1: Strip Markdown Links [anchor](url) -> anchor
    link_stripped = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] == "[":
            # Look for closing anchor bracket
            close_bracket = text.find("]", i)
            if (
                close_bracket != -1
                and close_bracket + 1 < n
                and text[close_bracket + 1] == "("
            ):
                close_paren = text.find(")", close_bracket + 2)
                if close_paren != -1:
                    # Append anchor text only, skip url payload
                    link_stripped.append(text[i + 1 : close_bracket])
                    i = close_paren + 1
                    continue
        link_stripped.append(text[i])
        i += 1

    text_no_links = "".join(link_stripped)

    # Step 2: Line-by-line structural filtering (Headers and List items)
    cleaned_lines = []
    for line in text_no_links.splitlines():
        # Remove leading whitespace to inspect structural tokens
        stripped_start = line.lstrip()
        if not stripped_start:
            cleaned_lines.append("")
            continue

        # Strip Headers (e.g., #, ##, ###)
        if stripped_start.startswith("#"):
            idx = 0
            while idx < len(stripped_start) and stripped_start[idx] == "#":
                idx += 1
            # Ensure it is a valid markdown header variant followed by whitespace
            if idx < len(stripped_start) and stripped_start[idx].isspace():
                line = stripped_start[idx:].lstrip()
                stripped_start = line

        # Strip Nested List Markers (-, *, + followed by space)
        if (
            len(stripped_start) >= 2
            and stripped_start[0] in ("-", "*", "+")
            and stripped_start[1].isspace()
        ):
            line = stripped_start[2:].lstrip()

        cleaned_lines.append(line)

    intermediate_text = "\n".join(cleaned_lines)

    # Step 3: Strip inline text-formatting characters (*, _, `, ~)
    final_chars = []
    exclude_chars = {"*", "_", "`", "~"}
    for char in intermediate_text:
        if char not in exclude_chars:
            final_chars.append(char)

    return "".join(final_chars).strip().lower()


def compute_md5(text: str) -> str:
    """Generates MD5 hash from normalized content for exact match filtering."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def setup_database(con: duckdb.DuckDBPyConnection) -> None:
    """Initializes schema for incremental state tracking."""
    con.execute(SQLQueries.CREATE_TABLE)


def sync_target_directory(con: duckdb.DuckDBPyConnection, target_path: Path) -> None:
    """Ingests changed files using mtime validation and purges deleted records."""
    db_paths = [r[0] for r in con.execute(SQLQueries.SELECT_ALL_FILEPATHS).fetchall()]

    for path_str in db_paths:
        if not Path(path_str).exists():
            con.execute(SQLQueries.DELETE_BY_FILEPATH, [path_str])
            print(f"Removed missing file from tracking: {path_str}")

    for file in target_path.glob("**/*.md"):
        try:
            current_mtime = file.stat().st_mtime
            row = con.execute(
                SQLQueries.SELECT_MTIME_BY_FILEPATH, [str(file)]
            ).fetchone()

            if row and row[0] == current_mtime:
                continue

            raw_text = file.read_text(encoding="utf-8")
            cleaned = clean_markdown(raw_text)
            md5_str = compute_md5(cleaned)

            con.execute(
                SQLQueries.UPSERT_DOCUMENT,
                (str(file), raw_text, cleaned, md5_str, current_mtime),
            )
            print(f"Parsed/Updated track state: {file.name}")
        except Exception as e:
            print(f"Failed processing {file}: {e}")


def process_exact_duplicates(con: duckdb.DuckDBPyConnection) -> None:
    """Identifies exact content duplicates via MD5 and establishes a canonical base."""
    con.execute(SQLQueries.RESET_CANONICAL_FLAGS)
    hashes = con.execute(SQLQueries.SELECT_DISTINCT_MD5).fetchall()

    for (md5_val,) in hashes:
        matching_files = con.execute(
            SQLQueries.SELECT_FILEPATHS_BY_MD5, [md5_val]
        ).fetchall()
        canonical_file = matching_files[0][0]
        con.execute(SQLQueries.SET_CANONICAL_TRUE, [canonical_file])


def compute_semantic_vectors(contents: tuple[str]) -> np.ndarray:
    """Transforms raw text into dimensionally reduced LSA representation vectors."""
    vectorizer = TfidfVectorizer(
        stop_words="english", sublinear_tf=True, max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(contents)

    n_components = min(100, tfidf_matrix.shape[0] - 1)
    if n_components > 1:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        return svd.fit_transform(tfidf_matrix)

    return tfidf_matrix.toarray()


def cluster_semantic_vectors(
    conceptual_matrix: np.ndarray, threshold: float
) -> np.ndarray:
    """Executes agglomerative clustering based on cosine distance parameters."""
    norms = np.linalg.norm(conceptual_matrix, axis=1)
    zero_indices = np.where(norms == 0)[0]
    valid_indices = np.where(norms > 0)[0]

    labels = np.full(conceptual_matrix.shape[0], -1, dtype=int)
    max_label = 0

    if len(valid_indices) >= 2:
        valid_matrix = conceptual_matrix[valid_indices]
        distance_threshold = 1.0 - threshold

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

    if len(zero_indices) > 0:
        empty_cluster_id = max_label + 1
        labels[zero_indices] = empty_cluster_id
        print(
            f"Isolated {len(zero_indices)} empty/stopword files (Cluster #{empty_cluster_id})."
        )

    return labels


def resolve_cluster_centroids(
    con: duckdb.DuckDBPyConnection,
    filepaths: tuple[str, ...],
    labels: np.ndarray,
    conceptual_matrix: np.ndarray,
) -> None:
    """Assigns clusters and re-evaluates canonical documents using centroid proximity."""
    for filepath, label in zip(filepaths, labels):
        con.execute(SQLQueries.UPDATE_CLUSTER_ID, (int(label), filepath))

    print("Resolving final consolidation mappings...")
    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]

        if len(cluster_indices) <= 1:
            continue

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

        con.execute(SQLQueries.SET_CLUSTER_CANONICAL_FALSE, [cluster_paths])
        con.execute(SQLQueries.SET_CANONICAL_TRUE, [chosen_canonical])

    con.execute(SQLQueries.PROPAGATE_EXACT_DUPLICATE_CLUSTERS)


def execute_physical_file_ops(
    con: duckdb.DuckDBPyConnection, action: str, target_dir: Path
) -> None:
    """Applies the consolidation strategy to disk storage architecture."""
    if action == "report":
        return

    drop_files = [
        r[0] for r in con.execute(SQLQueries.SELECT_NON_CANONICAL_CLUSTERED).fetchall()
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


def generate_pipeline_report(con: duckdb.DuckDBPyConnection) -> None:
    """Generates stdout logging detailing data reduction state."""
    print("\n" + "=" * 50 + "\nPIPELINE REDUCTION SUMMARY REPORT\n" + "=" * 50)
    results = con.execute(SQLQueries.PIPELINE_REDUCTION_REPORT).fetchall()

    if not results:
        print(
            "No duplicate configurations or conceptual clusters exceeded threshold parameters."
        )
    else:
        for cid, retain, dupes, count in results:
            print(f"\n[Conceptual Cluster #{cid}] -> Group Count: {count} files")
            print(f"  [KEEP] --> {retain}")
            print(f"  [DROP] --> {dupes}")


def parse_arguments() -> argparse.Namespace:
    """Configures and evaluates command line interface flags."""
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
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
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

    print("Executing Stage 1: MD5 exact match consolidation...")
    process_exact_duplicates(con)

    unique_docs = con.execute(SQLQueries.SELECT_CANONICAL_DOCUMENTS).fetchall()

    if len(unique_docs) < 2:
        print("Insufficient unique documents to run semantic clustering steps.")
        con.close()
        return

    filepaths, contents = zip(*unique_docs)

    print(
        "Executing Stage 2: Transforming text via TF-IDF + Latent Semantic Analysis..."
    )
    conceptual_matrix = compute_semantic_vectors(contents)

    print("Computing matrix distances and fitting agglomerative tree cluster layer...")
    labels = cluster_semantic_vectors(conceptual_matrix, args.threshold)

    resolve_cluster_centroids(con, filepaths, labels, conceptual_matrix)
    generate_pipeline_report(con)
    execute_physical_file_ops(con, args.action, target_path)

    con.close()


if __name__ == "__main__":
    main()
