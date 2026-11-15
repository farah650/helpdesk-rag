# Backlog MoSCoW — HelpdeskRAG

# Must

| ID | User story | Critère d'acceptation | Taille | Dépend de |
| --- | --- | --- | --- | --- |
| US01 | En tant qu'équipe, je veux télécharger et nettoyer le dataset de tickets afin d'avoir un corpus fiable | CSV dans `data/raw/` ; filtre `language=en`, `answer` non vide, dédoublonnage ; marqueurs `<tel_num>` et `<name>` nettoyés ; échantillon de 20 lignes contrôlé ; effectifs notés dans le README | M | — |
| US02 | En tant qu'équipe, je veux indexer chaque ticket (dense, sparse, métadonnées) dans Qdrant afin de le retrouver | Collection avec vecteurs dense et sparse ; payload `queue`, `type`, `priority`, `language`, `tag_1` à `tag_4` ; nombre de points = nombre de tickets indexés | M | US01 |
| US03 | En tant qu'agent, je veux une recherche hybride filtrable afin de retrouver les tickets proches de mon problème | Fusion RRF ; une requête filtrée par `queue` ne renvoie que des tickets de cette file (test automatique) ; scores renvoyés | M | US02 |
| US04 | En tant qu'agent, je veux que les résultats soient réordonnés par un reranker afin d'avoir les plus pertinents en premier | Cross-encoder appliqué aux candidats de la fusion ; hit rate top 3 mesuré avant et après reranking | M | US03 |
| US05 | En tant qu'agent, je veux une réponse (cause probable, actions, sources) afin de résoudre l'incident sans inventer | Sortie validée par un modèle Pydantic v2 ; 100 % des réponses citent au moins une source (test automatique) | M | US04 |
| US06 | En tant qu'agent, je veux voir la réponse s'afficher en streaming afin de ne pas attendre la fin | Endpoint FastAPI en SSE ; latence du premier token < 3 s mesurée | M | US05 |
| US07 | En tant qu'agent, je veux une interface simple afin de poser ma question et lire les sources | Interface Next.js : champ de question, réponse streamée, sources cliquables, avertissement « suggestion à valider, ne remplace pas la procédure officielle » | M | US06 |
| US08 | En tant qu'agent, je veux accepter ou rejeter la réponse afin que rien ne parte sans mon accord | Boutons accepter et rejeter ; aucun envoi sans clic ; décision enregistrée dans PostgreSQL (ticket, réponse, décision, horodatage) | M | US07 |
| US09 | En tant qu'équipe, nous voulons un jeu de test de 50 questions afin d'évaluer le système | 50 tickets mis de côté (hors index), questions reformulées, 3 groupes (identifiants, paraphrases, filtres), ticket et réponse attendus dans `eval/` | M | US01 |
| US10 | En tant qu'équipe, nous voulons comparer sparse, dense, hybride et hybride + reranker afin de justifier l'architecture | Une commande produit le tableau du hit rate top 3 par configuration et par groupe de questions | M | US04, US09 |
| US11 | En tant qu'équipe, nous voulons mesurer la fidélité des réponses afin de contrôler les hallucinations | Faithfulness Ragas ≥ 0,8 sur les 50 questions ; script reproductible | M | US05, US09 |
| US12 | En tant qu'équipe, nous voulons lancer la pile en une commande afin que le projet soit reproductible | `docker compose up` démarre Qdrant, PostgreSQL 16 et l'API ; README avec les étapes | S | — |

## Should

| ID | User story | Critère d'acceptation | Taille | Dépend de |
| --- | --- | --- | --- | --- |
| US13 | En tant qu'agent, je veux aussi des procédures, pas seulement des tickets, afin d'avoir la démarche officielle | Documentation officielle et Server Fault / Super User ingérées après vérification de leur licence ; sources citées avec le nom du document et la section | L | US02 |
| US14 | En tant qu'équipe, nous voulons comparer chunking sémantique et taille fixe afin de choisir le découpage | Context precision et hit rate mesurés pour les deux méthodes sur les documents de la base de connaissances | M | US13 |
| US15 | En tant qu'agent, je veux que le système refuse de répondre sans preuve afin d'éviter une réponse inventée | Abstention ≥ 90 % sur 10 questions hors périmètre ; message indiquant qu'aucune source fiable n'a été trouvée | M | US05 |
| US16 | En tant qu'équipe, nous voulons tracer chaque requête afin de diagnostiquer latence et contexte | Trace Langfuse par requête avec latence, passages retrouvés et réponse ; latence du premier token lue dans Langfuse | M | US06 |
| US17 | En tant qu'équipe, nous voulons une intégration continue afin de détecter les régressions | GitHub Actions exécute ruff et pytest à chaque push ; pipeline vert sur la branche principale | S | US12 |
| US18 | En tant qu'agent, je veux filtrer par file, type et priorité dans l'interface afin de restreindre la recherche | Sélecteurs `queue`, `type`, `priority` ; résultats limités aux valeurs choisies | S | US03, US07 |

## Could et Won't

**Could** : utiles si le temps le permet, sans risque pour le MVP.

| ID | User story | Critère d'acceptation | Taille | Dépend de |
| --- | --- | --- | --- | --- |
| US19 | En tant que responsable, je veux une suggestion de `queue` et de `priority` afin de router le ticket (sujet 09) | Précision mesurée sur les 50 tickets mis de côté | M | US02 |
| US20 | En tant que responsable, je veux voir les incidents récurrents afin de proposer une automatisation | Regroupement de tickets similaires avec effectifs par groupe | M | US02 |
| US21 | En tant qu'agent, je veux traiter aussi les tickets en allemand afin de couvrir toute la base | Modèle d'embedding multilingue ; évaluation séparée sur `language=de` | L | US10 |
| US22 | En tant qu'agent, je veux retrouver mes requêtes passées afin de reprendre un dossier | Liste des requêtes et décisions lue depuis PostgreSQL | S | US08 |

**Won't** : hors périmètre du MVP, assumé dès maintenant.

| ID | Élément exclu | Raison |
| --- | --- | --- |
| W1 | Fine-tuning d'un modèle 70B | Coût de calcul sans rapport avec l'objectif RAG |
| W2 | Facturation SaaS multi-tenant | Hors sujet pour un prototype académique |

## 
