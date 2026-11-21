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
from pydoc import text
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
    from langchain_text_splitters import RecursiveCharacterTextSplitter   
    from tqdm import tqdm                                                 
except ImportError as e:
    logger.error(f"Bibliothèque manquante : {e}")
    logger.error("Veuillez installer les dépendances : pip install qdrant-client sentence-transformers langchain-text-splitters tqdm")
    sys.exit(1)

def env(name: str) -> str:
    """Lit une variable dans .env. Pas de valeur par défaut : si elle manque, on s'arrête."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        logger.error(f"Variable '{name}' absente du fichier .env")
        sys.exit(1)
    return value.strip()

# --- Lues dans .env ---
QDRANT_HOST = env("QDRANT_HOST")
QDRANT_PORT = int(env("QDRANT_REST_PORT"))
QDRANT_VECTOR_SIZE = int(env("QDRANT_VECTOR_SIZE"))
COLLECTION_NAME = env("QDRANT_COLLECTION_NAME")
EMBEDDING_MODEL_NAME = env("EMBEDDING_MODEL_NAME")
EMBEDDING_DEVICE = env("EMBEDDING_DEVICE")

# Stratégie de découpage (stratégie A)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MIN_CHUNK_CHARS = 30   # un fragment plus court (ex. un titre seul) est fusionné avec son voisin
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

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

def _clean(v):
    return None if pd.isna(v) else v  #convertit les valeurs vides (NaN) en None, car un NaN fait planter l’envoi vers Qdrant

class IngestionEngine:
    """Moteur d'ingestion et d'indexation vectorielle industrielle."""

    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        logger.info(f"Connexion à la base vectorielle Qdrant sur {host}:{port}...")
        self.client = QdrantClient(host=host, port=port)
        logger.info(f"Chargement du modèle d'embeddings : {EMBEDDING_MODEL_NAME}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
        self.vector_dim = self.embedder.get_sentence_embedding_dimension()
        logger.info(f"Modèle prêt. Dimension des vecteurs : {self.vector_dim}")
         # le découpeur est créé une seule fois
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap, separators=SEPARATORS, length_function=len
        )
        logger.info(f"Découpage : RecursiveCharacterTextSplitter (chunk_size={chunk_size}, overlap={overlap})")

    def create_collection_if_not_exists(self, collection_name: str = COLLECTION_NAME, recreate: bool = False):
        """Crée la collection avec index HNSW et métrique Cosine."""
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if exists and recreate:                                  
            logger.info(f"Suppression de la collection '{collection_name}' (--recreate)...")
            self.client.delete_collection(collection_name)
            exists = False

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

    def chunk_text(self, text: str) -> List[str]:
        """Stratégie A : RecursiveCharacterTextSplitter (512 / 64)."""
        pieces = [p.strip().lstrip(".?!;, ").strip() for p in self.splitter.split_text(text)]
        chunks: List[str] = []
        for p in pieces:
            if not p:
                continue
            if chunks and len(chunks[-1]) < MIN_CHUNK_CHARS:      # fusionne un fragment trop court
                chunks[-1] = f"{chunks[-1]}\n\n{p}"
            else:
                chunks.append(p)
        if len(chunks) > 1 and len(chunks[-1]) < MIN_CHUNK_CHARS:
            last = chunks.pop()
            chunks[-1] = f"{chunks[-1]}\n\n{last}"
        return chunks

    def ingest_dataframe(self, df, source, collection_name=COLLECTION_NAME, batch_size: int = 256):
        ids, texts, payloads = [], [], []
        for _, row in df.iterrows():
            for idx, chunk in enumerate(self.chunk_text(f"{row['subject']}\n\n{row['body']}")):
                ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{row['ticket_id']}-{idx}")))
                texts.append(chunk)
                payloads.append({
                    "doc_id": str(row["ticket_id"]), "source": source, "page": 1,
                    "category": _clean(row["queue"]), "content": chunk, "chunk_index": idx,
                    "subject": row["subject"], "answer": row["answer"],
                    "type": _clean(row["type"]), "priority": _clean(row["priority"]), 
                    "language": _clean(row["language"]),
                })

        logger.info(f"{len(df)} tickets -> {len(ids)} chunks à vectoriser")

        for i in tqdm(range(0, len(ids), batch_size), desc="Ingestion", unit="lot"):
            vectors = self.embedder.encode(texts[i:i + batch_size], batch_size=64,
                                       normalize_embeddings=True, show_progress_bar=False)
            self.client.upsert(collection_name, points=qmodels.Batch(
                ids=ids[i:i + batch_size], vectors=vectors.tolist(), payloads=payloads[i:i + batch_size]))

        total = self.client.count(collection_name, exact=True).count
        logger.info(f"{len(ids)} chunks insérés pour {len(df)} tickets. Points dans '{collection_name}' : {total}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/tickets_clean.csv")
    ap.add_argument("--collection", default=COLLECTION_NAME)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)         
    ap.add_argument("--chunk-overlap", type=int, default=None)            
    ap.add_argument("--recreate", action="store_true")                    
    args = ap.parse_args()

    overlap = args.chunk_overlap if args.chunk_overlap is not None else (
        CHUNK_OVERLAP if args.chunk_size == CHUNK_SIZE else args.chunk_size // 8)
    
    df = load_tickets(args.csv)
    if args.limit:
        df = df.head(args.limit)

    engine = IngestionEngine(chunk_size=args.chunk_size, overlap=overlap)
    engine.create_collection_if_not_exists(args.collection, recreate=args.recreate)
    engine.ingest_dataframe(df, Path(args.csv).name, args.collection)