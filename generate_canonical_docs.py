import argparse
import re
from pathlib import Path
import duckdb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def fetch_cluster_data(db_path: str) -> dict[int, list[dict]]:
    """Queries DuckDB and groups source paths and content by cluster_id."""
    con = duckdb.connect(database=db_path, read_only=True)
    query = """
        SELECT cluster_id, filepath, raw_content 
        FROM documents 
        WHERE cluster_id IS NOT NULL AND raw_content IS NOT NULL
    """
    results = con.execute(query).fetchall()
    con.close()

    clusters = {}
    for cluster_id, filepath, raw_content in results:
        clusters.setdefault(cluster_id, []).append({
            "filepath": filepath,
            "content": raw_content
        })
    return clusters


def split_into_sentences(text: str) -> list[str]:
    """Splits a block of text into individual sentences using regex boundaries."""
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def extract_top_keywords(text_corpus: list[str], top_n: int = 3) -> list[str]:
    """Extracts the highest scoring TF-IDF terms across a corpus to use as metadata."""
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(text_corpus)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_scores.argsort()[::-1][:top_n]
        return [feature_names[i] for i in top_indices]
    except ValueError:
        return ["semantic", "summary", "cluster"]


def synthesize_canonical_content(sources: list[dict], num_sentences: int) -> tuple[str, str]:
    """Generates a new markdown document with centroid-ranked sentences and source wikilinks."""
    raw_documents = [s["content"] for s in sources]
    
    # Step 1: Tokenize all source documents into a unified sentence pool
    sentence_pool = []
    for doc in raw_documents:
        sentence_pool.extend(split_into_sentences(doc))

    sentence_pool = list(set(sentence_pool))

    # Extract keywords early to define document metadata and file slug naming
    keywords = extract_top_keywords(raw_documents, top_n=3)
    title = " ".join(keywords).title()
    
    # Generate clean, alphanumeric file name slug
    title_slug = "_".join(keywords).lower().replace(" ", "_")
    title_slug = re.sub(r"[^a-z0-9_]", "", title_slug)

    # Base text synthesis logic block
    if len(sentence_pool) <= num_sentences:
        selected_sentences = sentence_pool
    else:
        # Step 2: Vectorize sentences using TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english")
        sentence_vectors = vectorizer.fit_transform(sentence_pool)
        
        # Step 3: Compute the Global Cluster Centroid Vector
        cluster_centroid = np.asarray(sentence_vectors.mean(axis=0))

        # Step 4: Calculate Cosine Proximity to Centroid
        sentence_norms = np.linalg.norm(sentence_vectors.toarray(), axis=1)
        centroid_norm = np.linalg.norm(cluster_centroid)

        if centroid_norm == 0:
            selected_sentences = sentence_pool[:num_sentences]
        else:
            sentence_norms[sentence_norms == 0] = 1.0
            dot_products = np.dot(sentence_vectors.toarray(), cluster_centroid.T).flatten()
            similarities = dot_products / (sentence_norms * centroid_norm)

            # Step 5: Extract and order the top-ranking sentences chronologically
            top_indices = similarities.argsort()[::-1][:num_sentences]
            top_indices.sort()
            selected_sentences = [sentence_pool[idx] for idx in top_indices]

    # Step 6: Format structural Markdown output
    markdown_document = [
        f"# Authoritative Canonical Document: {title}",
        f"\n> **System Note:** This file was synthetically generated using a centroid-proximity vector model across cluster source components.",
        "\n## Core Concepts",
    ]

    for sentence in selected_sentences:
        markdown_document.append(f"{sentence}\n")

    # Step 7: Append references using standard Wikilink format
    markdown_document.append("\n## Source References")
    wikilinks = [f"* [[{Path(s['filepath']).name}]]" for s in sources]
    markdown_document.append("\n".join(wikilinks))

    return "\n\n".join(markdown_document), title_slug


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize Authoritative Canonical Markdown Files with Source Wikilinks."
    )
    parser.add_argument(
        "--db", type=str, default="markdown_analysis.db", help="DuckDB state cache database path."
    )
    parser.add_argument(
        "--out-dir", type=str, default="./canonical_generated", help="Directory where generated files are saved."
    )
    parser.add_argument(
        "--length", type=int, default=7, help="Number of semantic sentences to include in generated file."
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Target database '{args.db}' does not exist. Run the deduplication script first.")
        return

    output_path = Path(args.out_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Loading conceptual clusters from database: {db_path}...")
    clusters = fetch_cluster_data(args.db)

    print(f"Synthesizing {len(clusters)} canonical documents...")
    for cluster_id, sources in clusters.items():
        print(f" -> Processing Generation Phase for Cluster #{cluster_id} ({len(sources)} source assets)...")
        
        # Unpack synthesized payload along with its generated clean title slug
        synthetic_markdown, title_slug = synthesize_canonical_content(sources, args.length)
        
        # Construct explicit naming layout containing cluster identifier and title elements
        file_name = f"canonical_cluster_{cluster_id}_{title_slug}.md" if title_slug else f"canonical_cluster_{cluster_id}.md"
        target_file = output_path / file_name
        
        target_file.write_text(synthetic_markdown, encoding="utf-8")
        print(f"    [SAVED] Created synthesized canonical target: {target_file}")

    print("\nGeneration completely successful. Check your output directory.")


if __name__ == "__main__":
    main()
