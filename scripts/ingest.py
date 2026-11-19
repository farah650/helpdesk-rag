"""
Pipeline d'Ingestion & Vectorisation — Projet Fédérateur AI-BD_BI
Spécialité IA & Big Data / BI — Promotion Terminale (M2)

Fonctionnalités :
1. Découpage textuel sémantique ou récursif (RecursiveCharacterTextSplitter)
2. Extraction et enrichissement des métadonnées (Payload JSON)
3. Vectorisation par lots (Batching d'embeddings)
4. Initialisation de la collection Qdrant (HNSW + Cosine)
5. Insertion des points vectoriels avec gestion de persistance
"""

import os
import sys
import uuid
import logging
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import argparse
# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IngestPipeline")

# Tentative d'import des bibliothèques requises
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logger.error(f"Bibliothèque manquante : {e}")
    logger.error("Veuillez installer les dépendances : pip install qdrant-client sentence-transformers")
    sys.exit(1)


# Configuration par défaut (peut être surchargée par les variables d'environnement)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_REST_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_enterprise_docs")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2") # Remplacer par BAAI/bge-m3 en GPU

REQUIRED_COLS = ["ticket_id", "subject", "body", "answer", "type", "queue", "priority", "language"]


def load_tickets(csv_path: str) -> pd.DataFrame:
    """Charge le CSV nettoyé et vérifie son schéma."""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path} : {missing}")
    for c in ["subject", "body", "answer"]:
        df[c] = df[c].fillna("").astype(str)
    logger.info(f"{len(df)} tickets chargés depuis {csv_path}")
    return df

class IngestionEngine:
    """Moteur d'ingestion et d'indexation vectorielle industrielle."""

    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT):
        logger.info(f"Connexion à la base vectorielle Qdrant sur {host}:{port}...")
        self.client = QdrantClient(host=host, port=port)
        logger.info(f"Chargement du modèle d'embeddings : {EMBEDDING_MODEL_NAME}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.vector_dim = self.embedder.get_sentence_embedding_dimension()
        logger.info(f"Modèle prêt. Dimension des vecteurs : {self.vector_dim}")

    def create_collection_if_not_exists(self, collection_name: str = COLLECTION_NAME):
        """Crée la collection avec index HNSW et métrique Cosine."""
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            logger.info(f"Création de la collection '{collection_name}' (dim={self.vector_dim}, metric=Cosine)...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_dim,
                    distance=qmodels.Distance.COSINE
                ),
                hnsw_config=qmodels.HnswConfigDiff(
                    m=16,               # Nombre de liens par nœud dans le graphe
                    ef_construct=100    # Précision lors de la construction
                )
            )
            logger.info("Collection créée avec succès.")
        else:
            logger.info(f"La collection '{collection_name}' existe déjà.")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Découpage récursif simple par paragraphes et phrases."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(current_chunk) + len(p) <= chunk_size:
                current_chunk += " " + p if current_chunk else p
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = p

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def ingest_dataframe(self, df, source, collection_name=COLLECTION_NAME):
        ids, texts, payloads = [], [], []
        for _, row in df.iterrows():
            for idx, chunk in enumerate(self.chunk_text(f"{row['subject']}\n\n{row['body']}")):
                ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{row['ticket_id']}-{idx}")))
                texts.append(chunk)
                payloads.append({
                    "doc_id": str(row["ticket_id"]), "source": source, "page": 1,
                    "category": row["queue"], "content": chunk, "chunk_index": idx,
                    "subject": row["subject"], "answer": row["answer"],
                    "type": row["type"], "priority": row["priority"], "language": row["language"],
                })

        vectors = self.embedder.encode(texts, batch_size=64, normalize_embeddings=True,
                                       show_progress_bar=True)
        for i in range(0, len(ids), 64):
            self.client.upsert(collection_name, points=qmodels.Batch(
                ids=ids[i:i + 64], vectors=vectors[i:i + 64].tolist(), payloads=payloads[i:i + 64]))
        logger.info(f"{len(ids)} chunks insérés pour {len(df)} tickets")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/tickets_clean.csv")
    ap.add_argument("--collection", default=COLLECTION_NAME)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    df = load_tickets(args.csv)
    if args.limit:
        df = df.head(args.limit)

    engine = IngestionEngine()
    engine.create_collection_if_not_exists(args.collection)
    engine.ingest_dataframe(df, Path(args.csv).name, args.collection)