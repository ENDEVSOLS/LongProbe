import chromadb

def populate():
    # Initialize a local persistent Chroma DB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="default")

    # Add dummy documents
    collection.add(
        documents=[
            "The refund policy allows for full refunds within 30 days of purchase. No questions asked.",
            "Our enterprise payment terms are net 60 days.",
            "Data is encrypted at rest using AES-256 and in transit using TLS 1.3.",
            "Standard shipping takes 5-7 business days."
        ],
        ids=["doc_refund_1", "doc_billing_1", "doc_security_1", "doc_shipping_1"],
        metadatas=[{"source": "legal"}, {"source": "finance"}, {"source": "tech"}, {"source": "logistics"}]
    )
    print("Successfully populated Chroma DB with dummy data.")

if __name__ == "__main__":
    populate()
