import os
from sentence_transformers import SentenceTransformer

DATA_DIR = "data/documents"

def load_faqs(data_dir):
    faqs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            faqs.append({
                "id": fname,
                "text": text
            })
    return faqs


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    faqs = load_faqs(DATA_DIR)
    texts = [faq["text"] for faq in faqs]

    embeddings = model.encode(texts, show_progress_bar=True)

    # TEMP: print to verify
    print(f"Loaded {len(faqs)} FAQs")
    print(f"Embedding shape: {embeddings[0].shape}")

    # TODO: store embeddings in Endee
    # for faq, vector in zip(faqs, embeddings):
    #     endee_client.store(
    #         id=faq["id"],
    #         vector=vector,
    #         metadata={"text": faq["text"]}
    #     )

if __name__ == "__main__":
    main()
