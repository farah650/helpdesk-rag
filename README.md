# helpdesk-rag
# Fiche de cadrage — HelpdeskRAG&#32;

## Équipe

Farah Ben Said\
Amal Balti\
Aymen Reghioui\
Ahmed Hajhassen

## Cas d'usage

HelpdeskRAG est un assistant RAG qui aide un agent support IT à résoudre un incident : il retrouve les tickets déjà résolus et les sections de procédures pertinentes, puis génère en streaming une réponse sourcée que l'agent valide ou rejette.

Exemple : « A user gets error 403 on the portal after resetting their password. What should I do? » La réponse donne la cause probable, les actions proposées et ses sources (tickets et procédure).

## Problème

Les équipes support perdent du temps à chercher dans des historiques de tickets et des procédures dispersés, et un incident déjà résolu est souvent retraité de zéro. L'agent a besoin d'une réponse vérifiable, avec sa source citée, pas d'une réponse inventée.

## Cible et valeur

- **Utilisateurs** : agents support N1 et N2.
- **Clients potentiels** : DSI, centres de services, sociétés d'infogérance.
- **Gains visés** : moins de temps de recherche par ticket et moins de réaffectations.
- **perspective** :un agent qui lit les e-mails entrants, les traite et y répond automatiquement.

Le MVP ne mesure pas ces gains sur de vrais agents. Il les approche par des indicateurs de qualité : part des questions où le ticket attendu apparaît dans le top 3, et part des réponses fidèles aux sources.

## Corpus et volumétrie

| Corpus | Source | Volume |
| --- | --- | --- |
| Historique de tickets | [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (Hugging Face), CC BY-NC 4.0 | 61 765 tickets au total (anglais et allemand) ; \[N\] après filtre `language=en`, `answer` non vide et dédoublonnage ; \[N\_index\] indexés |
| Jeu de test (non indexé) | 50 tickets mis de côté avec résolution attendue | 50 questions |

Colonnes utilisées : `subject`, `body`, `answer`, `type`, `queue`, `priority`, `language`, `tag_1` à `tag_4`.

- Un ticket = un chunk (`subject` + `body` + `answer`), avec les colonnes ci-dessus en métadonnées filtrables.
- Le dataset ne contient aucune procédure : c'est la base de connaissances qui les apporte, d'où la comparaison de chunking sur ce corpus-là.
- Les donneés de test sont reformulées à partir des tickets mis de côté , pour mesurer le hybride par type de requête.

## Retrieval et architecture

- **Dense** (embeddings) : comprend le sens des reformulations.
- **Sparse** (BM25 ou SPLADE) : retrouve les termes et codes exacts (« 403 », « VPN »).
- **Filtres** de métadonnées sur `queue`, `type`, `priority`, tags et sur le type de source (ticket ou procédure).
- **Fusion** RRF, puis reranker cross-encoder.
- **Stockage** : Qdrant, vecteurs dense et sparse dans la même collection (à confirmer dans la doc de la version utilisée).
- **Chemin d'une requête** : React, passerelle FastAPI, embeddings dense et sparse, Qdrant, reranker, LLM (API ou vLLM), réponse streamée. PostgreSQL 16 garde l'historique et les décisions de l'agent ; Langfuse ou OTel tracent chaque étape.

## Stack

FastAPI + Pydantic v2 · Qdrant · PostgreSQL 16 (historique et DTO) · LlamaIndex (ou LangChain) · reranker cross-encoder · LLM par API (ou vLLM) · Ragas + Langfuse · Next.js (React) avec SSE · Docker Compose · GitHub Actions (ruff + pytest).

## 

## 
