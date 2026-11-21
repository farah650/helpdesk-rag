"""
chunking.py : choix de la stratégie de découpage 
======================================================================

LES DEUX STRATÉGIES
  A) RecursiveCharacterTextSplitter  (chunks de 512 caractères, chevauchement 64)
     -> pour des textes longs avec sous-titres. On cherche et on renvoie le même morceau.
  B) Parent-Document Retriever       (petits chunks de 128, parents de 1024)
     -> pour des corpus réglementaires (juridiques, lois...). On cherche sur de petits morceaux,
        puis on renvoie le "parent" (1024 car.) pour donner plus de contexte au LLM.

MÉTHODE
  1. Mesurer la longueur de nos tickets (sujet + corps).
  2. Appliquer A et B sur un échantillon de 5000 tickets (sur 40 280).
  3. Comparer : nombre de vecteurs, taille des morceaux, tickets restant entiers.
  4. Appliquer une règle simple : B seulement si le corpus est réglementaire ET long.
  
UTILISATION
  python scripts/chunking.py                     # analyse sur 5000 tickets

"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    sys.exit("Bibliothèque manquante : pip install langchain-text-splitters")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # accents corrects sous Windows

# --------------------------------------------------------------------------- #
# Paramètres imposés par le sujet
# --------------------------------------------------------------------------- #
CHUNK_SIZE, CHUNK_OVERLAP = 512, 64      # stratégie A
PARENT_SIZE, PARENT_OVERLAP = 1024, 0    # stratégie B : parents
CHILD_SIZE, CHILD_OVERLAP = 128, 16      # stratégie B : petits chunks
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

REQUIRED_COLS = ["ticket_id", "subject", "body"]  # seules colonnes utiles au découpage

# Mots qui indiquent un texte réglementaire (articles de loi, RGPD, directives...)
LEGAL_RE = re.compile(
    r"(?:article\s+\d+|art\.\s*\d+|§|clause\s+\d+|\bregulation\b|\bdirective\b|\bgdpr\b|\brgpd\b|"
    r"\bdsgvo\b|\bcompliance\b|\bgesetz\b|\bverordnung\b)",
    re.IGNORECASE,
)


def make_splitter(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=SEPARATORS, length_function=len
    )


splitter_a = make_splitter(CHUNK_SIZE, CHUNK_OVERLAP)
splitter_parent = make_splitter(PARENT_SIZE, PARENT_OVERLAP)
splitter_child = make_splitter(CHILD_SIZE, CHILD_OVERLAP)


# --------------------------------------------------------------------------- #
# Les deux stratégies
# --------------------------------------------------------------------------- #
def _clean(pieces: List[str], min_len: int = 30) -> List[str]:
    """Enlève le séparateur resté en début de morceau (ex. '. Texte') et fusionne
    les fragments trop courts (< 30 car., ex. un titre seul) avec leur voisin."""
    out: List[str] = []
    for p in (x.strip().lstrip(".?!;, ").strip() for x in pieces):
        if not p:
            continue
        if out and len(out[-1]) < min_len:
            out[-1] = f"{out[-1]}\n\n{p}"
        else:
            out.append(p)
    if len(out) > 1 and len(out[-1]) < min_len:
        last = out.pop()
        out[-1] = f"{out[-1]}\n\n{last}"
    return out


def strategy_a(text: str) -> List[str]:
    """A : liste de chunks (512 / 64)."""
    return _clean(splitter_a.split_text(text))


def strategy_b(text: str) -> List[Dict]:
    """B : liste de parents (1024), chacun avec ses petits chunks (128)."""
    return [{"parent": p, "children": _clean(splitter_child.split_text(p))} for p in _clean(splitter_parent.split_text(text))]


# --------------------------------------------------------------------------- #
# Données et mesures
# --------------------------------------------------------------------------- #
def load_tickets(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Colonnes manquantes dans {csv_path} : {missing}")
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["text"] = (df["subject"].str.strip() + "\n\n" + df["body"].str.strip()).str.strip()
    return df[df["text"] != ""].reset_index(drop=True)


def measure(df: pd.DataFrame) -> Dict[str, float]:
    lens = df["text"].str.len()
    a_counts, a_lens, p_counts, c_counts, c_lens = [], [], [], [], []
    for text in df["text"]:
        chunks = strategy_a(text)
        a_counts.append(len(chunks))
        a_lens += [len(c) for c in chunks]
        parents = strategy_b(text)
        kids = [c for p in parents for c in p["children"]]
        p_counts.append(len(parents))
        c_counts.append(len(kids))
        c_lens += [len(c) for c in kids]
    a_counts, p_counts, c_counts = pd.Series(a_counts), pd.Series(p_counts), pd.Series(c_counts)
    return {
        "median": lens.median(),
        "p90": lens.quantile(0.90),
        "over_1024": (lens > PARENT_SIZE).mean(),
        "legal": df["text"].str.contains(LEGAL_RE).mean(),
        "a_vectors": a_counts.sum(),
        "a_avg_len": sum(a_lens) / len(a_lens),
        "a_single": (a_counts == 1).mean(),
        "b_vectors": c_counts.sum(),
        "b_avg_len": sum(c_lens) / len(c_lens),
        "b_single_parent": (p_counts == 1).mean(),
    }


def decide(m: Dict[str, float]) -> str:
    """Règle : B seulement si le corpus est réglementaire (>= 30 %) ET long (>= 30 % > 1024 car.)."""
    return "B" if (m["legal"] >= 0.30 and m["over_1024"] >= 0.30) else "A"


# --------------------------------------------------------------------------- #
# Affichage
# --------------------------------------------------------------------------- #
def short(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "..."


def example_lines(row: pd.Series) -> List[str]:
    text = row["text"]
    chunks = strategy_a(text)
    parents = strategy_b(text)
    kids = [c for p in parents for c in p["children"]]
    lines = [f"Ticket {row['ticket_id']} ({len(text)} caractères) : {short(str(row['subject']), 60)}", ""]
    lines.append(f"  A -> {len(chunks)} morceau(x) :")
    for i, c in enumerate(chunks[:3]):
        lines.append(f"       [{i + 1}] {len(c)} car. : {short(c)}")
    lines.append(f"  B -> {len(parents)} parent(s) et {len(kids)} petits morceaux, par exemple :")
    for i, c in enumerate(kids[:3]):
        lines.append(f"       [{i + 1}] {len(c)} car. : {short(c)}")
    if len(kids) > 3:
        lines.append(f"       ... et {len(kids) - 3} autres")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Choix de la stratégie de découpage sur tickets_clean.csv")
    ap.add_argument("--csv", default="data/processed/tickets_clean.csv")
    ap.add_argument("--sample", type=int, default=5000, help="Nombre de tickets analysés (0 = tous)")
    ap.add_argument("--ticket-id", help="Exemple sur un ticket précis (sinon : ticket médian)")
    ap.add_argument("--seed", type=int, help="Fixe le tirage au hasard des 3 exemples (pour retrouver les mêmes)")
    ap.add_argument("--report", help="Fichier .md où sauvegarder le résultat")
    args = ap.parse_args()

    df = load_tickets(args.csv)
    total = len(df)
    if args.sample and args.sample < total:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)

    m = measure(df)
    winner = decide(m)

    if args.ticket_id:
        found = df[df["ticket_id"].astype(str) == str(args.ticket_id)]
        if found.empty:
            sys.exit(f"ticket_id {args.ticket_id} introuvable dans l'échantillon (essaie --sample 0)")
        rows = [found.iloc[0]]
    else:
        picked = df.sample(min(3, len(df)), random_state=args.seed)  # 3 tickets au hasard
        rows = [r for _, r in picked.iterrows()]
    example_block: List[str] = []
    for i, row in enumerate(rows):
        if i:
            example_block.append("")
        example_block += [f"   {line}" for line in example_lines(row)]

    bar = "=" * 66
    out = [
        bar,
        f" CHOIX DE LA STRATÉGIE DE DÉCOUPAGE   ({len(df)} tickets analysés sur {total})",
        bar,
        "",
        "1) NOS DONNÉES",
        f"   Longueur médiane d'un ticket       : {m['median']:.0f} caractères",
        f"   9 tickets sur 10 font moins de     : {m['p90']:.0f} caractères",
        f"   Tickets de plus de 1024 caractères : {m['over_1024'] * 100:.1f} %",
        f"   Tickets de type réglementaire      : {m['legal'] * 100:.1f} %",
        "",
        "2) COMPARAISON",
        f"   {'':34s}{'A : Recursive':>16s}{'B : Parent-Doc':>16s}",
        f"   {'':34s}{'512 / 64':>16s}{'128 / 1024':>16s}",
        f"   {'Vecteurs à stocker':34s}{m['a_vectors']:>16.0f}{m['b_vectors']:>16.0f}",
        f"   {'Taille moyenne d un morceau (car.)':34s}{m['a_avg_len']:>16.0f}{m['b_avg_len']:>16.0f}",
        f"   {'Tickets gardés en un seul morceau':34s}{m['a_single'] * 100:>15.0f}%{m['b_single_parent'] * 100:>15.0f}%",
        "",
        "3) EXEMPLES (3 tickets au hasard)" if not args.ticket_id else "3) EXEMPLE",
        *example_block,
        "",
        "4) DÉCISION",
        "   -> Stratégie A : RecursiveCharacterTextSplitter (chunk_size=512, chunk_overlap=64)"
        if winner == "A"
        else "   -> Stratégie B : Parent-Document Retriever (petits chunks 128, parents 1024)",
        bar,
    ]
    print("\n".join(out))

    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(out), encoding="utf-8")
        print(f"Résultat sauvegardé : {p}")


if __name__ == "__main__":
    main()