"""US01 : nettoyage du dataset Tobi-Bueck/customer-support-tickets (anglais + allemand).

Entrée  : data/raw/tickets_raw.csv
Sortie  : data/processed/tickets_clean.parquet (+ .csv pour ouvrir dans Excel)
"""
import re
import sys
from pathlib import Path

import pandas as pd

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/tickets_raw.csv")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["en", "de"]
# Colonnes supprimées : tag_1 à tag_8 + version (non utilisée dans le projet)
DROP = [f"tag_{i}" for i in range(1, 9)] + ["version"]
TEXT_COLS = ["subject", "body", "answer"]
KEEP = ["ticket_id", "subject", "body", "answer", "type", "queue", "priority", "language"]

# Marqueurs d'anonymisation (anglais + allemand), formes <...> et [...]
OPEN, CLOSE = r"[<\[]\s*", r"\s*[>\]]"
NAME_RE = re.compile(
    OPEN + r"(?:(?:your|customer|client|agent|support team|user|ihr|ihre|dein|deine|kunden)[\s_]+)?name" + CLOSE, re.I)
PHONE_RE = re.compile(
    OPEN + r"(?:tel(?:efon)?[.\s_-]*(?:num(?:mer|ber)?|nr\.?)|telefon|phone(?:[ _]number)?)" + CLOSE, re.I)
ACC_RE = re.compile(OPEN + r"(?:(?:acc(?:ount)?|konto)[.\s_-]*(?:num(?:mer|ber)?|nr\.?))" + CLOSE, re.I)
OTHER_RE = re.compile(r"<([a-z][a-z0-9_]{1,25})>", re.I)          # <email>, <tool_name>, <price>...
HTML_RE = re.compile(r"</?(?:br|body|p|div)\s*/?>", re.I)           # <br>, <br/>, </br>, </body>...
# salutation restée vide après suppression du nom : "Dear," / "Sehr geehrte/r," / "Hallo,"
GREETING_RE = re.compile(
    r"^(?:dear|hello|hi|hey|hallo|guten tag|liebe[rs]?|sehr geehrte[rs]?(?:/r)?(?:\[r\]|\(r\))?)\s*,\s*", re.I)


def clean_text(s) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\\n", " ")                              # "\n" écrit en toutes lettres dans les données
    s = HTML_RE.sub(" ", s)
    s = NAME_RE.sub(" ", s)                                 # noms anonymisés : supprimés
    s = PHONE_RE.sub("[phone]", s)                          # téléphone : jeton neutre
    s = ACC_RE.sub("[account number]", s)
    s = OTHER_RE.sub(lambda m: "[" + m.group(1).replace("_", " ") + "]", s)  # <email> -> [email]
    s = re.sub(r"\s+", " ", s)                              # espaces multiples
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)                  # "Hello , x" -> "Hello, x"
    s = re.sub(r"[,;:]+\s*([.!?])", r"\1", s)                # "out,." -> "out."
    s = re.sub(r"^[,;:\s]+", "", s)                         # ", we acknowledge" -> "we acknowledge"
    s = GREETING_RE.sub("", s)                              # salutation vide
    s = s.strip()
    return s[:1].upper() + s[1:] if s else s                # majuscule en début de texte


def main() -> None:
    df = pd.read_csv(RAW)
    n_raw = len(df)

    # 1. suppression des colonnes inutiles
    df = df.drop(columns=[c for c in DROP if c in df.columns])

    # 2. anglais + allemand
    df = df[df["language"].isin(LANGUAGES)].copy()
    n_lang = len(df)

    # 3. nettoyage du texte
    for c in TEXT_COLS:
        df[c] = df[c].map(clean_text)

    # 4. réponse et question exploitables (vides ou trop courtes écartées)
    df = df[(df["answer"].str.len() >= 15) & (df["body"].str.len() >= 15)]
    n_ans = len(df)

    # 5. dédoublonnage
    df = df.drop_duplicates(subset=TEXT_COLS).reset_index(drop=True)
    n_final = len(df)

    df.insert(0, "ticket_id", range(n_final))
    df = df[KEEP]
    df.to_parquet(PROC / "tickets_clean.parquet", index=False)
    df.to_csv(PROC / "tickets_clean.csv", index=False)

    print(f"Total brut                    : {n_raw}")
    print(f"language en + de              : {n_lang}")
    print(f"answer/body exploitables      : {n_ans}")
    print(f"après dédoublonnage (final)   : {n_final}")
    print("\nPar langue :\n", df["language"].value_counts().to_string())
    print(f"\nColonnes finales : {list(df.columns)}")
    print("\nRépartition queue :\n", df["queue"].value_counts().to_string())
    print("\nRépartition type :\n", df["type"].value_counts().to_string())
    left = df[TEXT_COLS].apply(
        lambda c: c.str.contains(
            r"(?i:<[a-z_/]+/?>|\[(?:(?:your|ihr|ihre|dein|customer) )?name\]|\[tel[ _.-]*(?:num|nr)\w*\.?\]|\[acc_num\])|\\n",
            regex=True)
    ).any().any()
    print("\nMarqueurs restants :", left)
    print("\nÉchantillon de 20 lignes à contrôler (tronqué à 110 caractères) :")
    for _, r in df.sample(20, random_state=42).iterrows():
        print(f"\n#{r.ticket_id} [{r.language}] [{r.queue}] {r.subject[:80]}")
        print("  Q:", r.body[:110])
        print("  R:", r.answer[:110])


if __name__ == "__main__":
    main()