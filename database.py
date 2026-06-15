from typing import Union
from pathlib import Path
import duckdb


class SQLQueries:
    """Isolates all raw SQL statements from the transactional database logic."""

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

    FETCH_CLUSTER_DATA = """
        SELECT cluster_id, filepath, raw_content 
        FROM documents 
        WHERE cluster_id IS NOT NULL AND raw_content IS NOT NULL
    """


class DatabaseManager:
    """Manages DuckDB state mutations and validates query execution environments."""

    def __init__(self, db_path: str):
        assert db_path, "Database path target string must be initialized."
        self.db_path = db_path
        self.con = None

    def __enter__(self):
        self.con = duckdb.connect(database=self.db_path)
        assert self.con is not None, "DuckDB engine context initialization failure."
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.con:
            self.con.close()

    def setup_schema(self) -> None:
        """Initializes structural table definitions for idempotent ingestion tracking."""
        assert self.con is not None, (
            "Database operation rejected: connection context closed or dead."
        )
        self.con.execute(SQLQueries.CREATE_TABLE)

    def get_all_filepaths(self) -> list[str]:
        """Retrieves all tracked asset paths from the database."""
        assert self.con is not None, (
            "Database operation rejected: connection context closed or dead."
        )
        return [
            r[0] for r in self.con.execute(SQLQueries.SELECT_ALL_FILEPATHS).fetchall()
        ]

    def delete_by_filepath(self, filepath: str) -> None:
        """Purges an unlinked tracking row by its unique file system path identifier."""
        assert self.con is not None, (
            "Database operation rejected: connection context closed or dead."
        )
        assert filepath, (
            "Cannot execute file unlinking query with missing or empty path payload."
        )
        self.con.execute(SQLQueries.DELETE_BY_FILEPATH, [filepath])

    def get_mtime_by_filepath(self, filepath: str) -> Union[float, None]:
        """Fetches the cached modification time scalar for delta synchronization checks."""
        assert self.con is not None, (
            "Database operation rejected: connection context closed or dead."
        )
        assert filepath, "Invalid query target: source filepath parameter required."
        row = self.con.execute(
            SQLQueries.SELECT_MTIME_BY_FILEPATH, [filepath]
        ).fetchone()
        return row[0] if row else None

    def upsert_document(
        self,
        filepath: str,
        raw_content: str,
        cleaned_content: str,
        md5_hash: str,
        mtime: float,
    ) -> None:
        """Executes an upsert mutation reset sequence on modified document files."""
        assert self.con is not None, (
            "Database operation rejected: connection context closed or dead."
        )
        assert filepath, (
            "Document tracking constraint violation: Target storage filepath parameter required."
        )
        assert md5_hash, "State mutation rejected: Calculated content hash is empty."
        assert mtime >= 0, (
            "Invalid transactional metadata: Modification metric cannot be negative."
        )

        self.con.execute(
            SQLQueries.UPSERT_DOCUMENT,
            (filepath, raw_content, cleaned_content, md5_hash, mtime),
        )

    def process_exact_duplicates(self) -> None:
        """Resolves structural base matches via raw MD5 matching and flags source targets."""
        assert self.con is not None, (
            "Database operation rejected: connection context closed or dead."
        )
        self.con.execute(SQLQueries.RESET_CANONICAL_FLAGS)
        hashes = self.con.execute(SQLQueries.SELECT_DISTINCT_MD5).fetchall()

        for (md5_val,) in hashes:
            matching_files = self.con.execute(
                SQLQueries.SELECT_FILEPATHS_BY_MD5, [md5_val]
            ).fetchall()
            assert len(matching_files) > 0, (
                "MD5 data integrity variance encountered: Hash mapping tracks empty pool."
            )

            canonical_file = matching_files[0][0]
            self.con.execute(SQLQueries.SET_CANONICAL_TRUE, [canonical_file])

    def get_canonical_documents(self) -> list[tuple[str, str]]:
        """Extracts text components of documents designated as unique exact matches."""
        assert self.con is not None, (
            "Database operation rejected: connection context closed or dead."
        )
        return self.con.execute(SQLQueries.SELECT_CANONICAL_DOCUMENTS).fetchall()

    def update_cluster_id(self, filepath: str, cluster_id: int) -> None:
        """Updates cluster identifier mappings across execution components."""
        assert self.con is not None, (
            "Database operation rejected: connection context closed or dead."
        )
        assert filepath, "Cluster serialization error: Path identifier cannot be empty."
        assert cluster_id >= -1, "Cluster structural index bounds exception."

        self.con.execute(SQLQueries.UPDATE_CLUSTER_ID, (cluster_id, filepath))

    def set_cluster_canonical(
        self, cluster_paths: list[str], canonical_path: str
    ) -> None:
        """Applies explicit Boolean flags establishing absolute canonical authority."""
        assert self.con is not None, (
            "Database operation rejected: connection context closed or dead."
        )
        assert cluster_paths, (
            "Consolidation batch list sequence must contain at least one string item."
        )
        assert canonical_path, (
            "Canonical target pointer target assignment cannot be empty."
        )
        assert canonical_path in cluster_paths, (
            "Logical conflict: Core canonical target must reside inside processing pool."
        )

        self.con.execute(SQLQueries.SET_CLUSTER_CANONICAL_FALSE, [cluster_paths])
        self.con.execute(SQLQueries.SET_CANONICAL_TRUE, [canonical_path])

    def propagate_exact_duplicate_clusters(self) -> None:
        """Cascades parent cluster labels down to secondary physical exact copies."""
        assert self.con is not None, (
            "Database operation rejected: connection context closed or dead."
        )
        self.con.execute(SQLQueries.PROPAGATE_EXACT_DUPLICATE_CLUSTERS)

    def get_non_canonical_clustered(self) -> list[str]:
        """Identifies target file instances mapped for relocation or removal routines."""
        assert self.con is not None, (
            "Database operation rejected: connection context closed or dead."
        )
        return [
            r[0]
            for r in self.con.execute(
                SQLQueries.SELECT_NON_CANONICAL_CLUSTERED
            ).fetchall()
        ]

    def get_pipeline_report(self) -> list[tuple]:
        """Runs summary metrics capturing group data reduction stats."""
        assert self.con is not None, (
            "Database operation rejected: connection context closed or dead."
        )
        return self.con.execute(SQLQueries.PIPELINE_REDUCTION_REPORT).fetchall()

    def fetch_cluster_data(self) -> dict[int, list[dict]]:
        """Aggregates all cluster data structural models for payload generation tasks."""
        assert self.con is not None, (
            "Database operation rejected: connection context closed or dead."
        )
        results = self.con.execute(SQLQueries.FETCH_CLUSTER_DATA).fetchall()

        clusters = {}
        for cluster_id, filepath, raw_content in results:
            assert cluster_id is not None, (
                "Logical invariant broken: Found untracked structural cluster context node."
            )
            clusters.setdefault(cluster_id, []).append(
                {"filepath": filepath, "content": raw_content}
            )
        return clusters
