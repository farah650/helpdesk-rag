from sentence_transformers import SentenceTransformer, util

m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Dimension      :", m.get_sentence_embedding_dimension())
print("max_seq_length :", m.max_seq_length)

en = "I want to report a security incident on cloud storage"
de = "Ich möchte einen Sicherheitsvorfall im Cloud-Speicher melden"
autre = "How do I change my billing address?"

v = m.encode([en, de, autre], normalize_embeddings=True)
print("en vs de (même sujet) :", round(float(util.cos_sim(v[0], v[1])), 3))
print("en vs autre sujet     :", round(float(util.cos_sim(v[0], v[2])), 3))