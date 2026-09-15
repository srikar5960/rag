from sentence_transformers import SentenceTransformer


class EmbeddingManager:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings.tolist()

    def generate_query_embedding(self, query):

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        return embedding.tolist()