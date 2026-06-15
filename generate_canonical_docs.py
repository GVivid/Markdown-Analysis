import argparse
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import data access layer adapter
from database import DatabaseManager


def split_into_sentences(text: str) -> list[str]:
    """Splits text into sentences using standard string iteration, filtering out fragments."""
    normalized_text = " ".join(text.split())
    
    sentences = []
    start_idx = 0
    text_len = len(normalized_text)
    
    for i, char in enumerate(normalized_text):
        if char in ".!?":
            if i == text_len - 1 or normalized_text[i + 1] == " ":
                sentences.append(normalized_text[start_idx : i + 1].strip())
                start_idx = i + 1
                
    if start_idx < text_len:
        remainder = normalized_text[start_idx:].strip()
        if remainder:
            sentences.append(remainder)
            
    return [s for s in sentences if len(s) > 10]


def extract_top_keywords(text_corpus: list[str], top_n: int = 3) -> list[str]:
    """Extracts the highest scoring TF-IDF terms across a corpus for metadata use."""
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
    """Generates a markdown document using sparse-matrix centroid ranking."""
    raw_documents = [s["content"] for s in sources]
    
    raw_sentence_pool = []
    for doc in raw_documents:
        raw_sentence_pool.extend(split_into_sentences(doc))

    seen = set()
    sentence_pool = [s for s in raw_sentence_pool if not (s in seen or seen.add(s))]

    keywords = extract_top_keywords(raw_documents, top_n=3)
    title = " ".join(keywords).title()
    
    raw_slug = "_".join(keywords).lower()
    title_slug = "".join(char for char in raw_slug if char.isalnum() or char == "_")

    if len(sentence_pool) <= num_sentences:
        selected_sentences = sentence_pool
    else:
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            sentence_vectors = vectorizer.fit_transform(sentence_pool)
            cluster_centroid = np.asarray(sentence_vectors.mean(axis=0))
            similarities = cosine_similarity(sentence_vectors, cluster_centroid).flatten()

            top_indices = similarities.argsort()[::-1][:num_sentences]
            top_indices.sort()
            selected_sentences = [sentence_pool[idx] for idx in top_indices]
            
        except ValueError:
            selected_sentences = sentence_pool[:num_sentences]

    markdown_document = [
        f"# Authoritative Canonical Document: {title}",
        "\n> **System Note:** This file was synthetically generated using a centroid-proximity vector model across cluster source components.",
        "\n## Core Concepts",
    ]

    for sentence in selected_sentences:
        markdown_document.append(f"{sentence}\n")

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
    with DatabaseManager(args.db) as db:
        clusters = db.fetch_cluster_data()

    if not clusters:
        print("No valid clustered document records found. Exiting.")
        return

    print(f"Synthesizing {len(clusters)} canonical documents...")
    for cluster_id, sources in clusters.items():
        print(f" -> Processing Generation Phase for Cluster #{cluster_id} ({len(sources)} source assets)...")
        
        synthetic_markdown, title_slug = synthesize_canonical_content(sources, args.length)
        
        file_name = f"canonical_cluster_{cluster_id}_{title_slug}.md" if title_slug else f"canonical_cluster_{cluster_id}.md"
        target_file = output_path / file_name
        
        target_file.write_text(synthetic_markdown, encoding="utf-8")
        print(f"    [SAVED] Created synthesized canonical target: {target_file}")

    print("\nGeneration completely successful. Check your output directory.")


if __name__ == "__main__":
    main()
