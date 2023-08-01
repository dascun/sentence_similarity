from sentence_similarity import sentence_similarity

class SentenceComparer:
    def __init__(self, model_name, embedding_type="cls_token_embedding", metric="cosine"):
        """Initialize the SentenceComparer with the specified NLP model."""
        self.model = sentence_similarity(model_name=model_name, embedding_type=embedding_type)
        self.metric = metric

    def compare_sentences(self, sentence_1, sentence_2):
        """Calculate the similarity score between two sentences using the initialized NLP model."""
        score = self.model.get_score(sentence_1, sentence_2, metric=self.metric)
        return f"Comparison Score between '{sentence_1}' and '{sentence_2}': {score}"

if __name__ == "__main__":
    model_1 = "sentence-transformers/all-MiniLM-L6-v2"
    model_2 = "sentence-transformers/all-mpnet-base-v2"
    model_3 = "sentence-transformers/paraphrase-MiniLM-L12-v2"
    model_4 = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    model_5 = "sentence-transformers/nli-mpnet-base-v2"

    models = [model_1, model_2, model_3, model_4, model_5]

    sentence_1 = "rivers woods and hills"
    sentence_2 = "streams forests and mountains"
    sentence_3 = "deserts sand and shrubs"

    for model_name in models:
        comparer = SentenceComparer(model_name=model_name)
        print(f"Now using model: '{model_name}'")
        print(comparer.compare_sentences(sentence_1=sentence_1, sentence_2=sentence_2))
        print(comparer.compare_sentences(sentence_1=sentence_1, sentence_2=sentence_3))
