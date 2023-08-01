from sentence_similarity import sentence_similarity

def compare_sentences(sentence_1=str, sentence_2=str, model_name=str, embedding_type="cls_token_embedding", metric="cosine") -> str:
    """Utilizes an NLP model that calculates the similarity between 
    two sentences or phrases."""

    model = sentence_similarity(model_name=model_name, embedding_type=embedding_type)
    score = model.get_score(sentence_1, sentence_2, metric=metric)
    return(f"Comparison Score between '{sentence_1}' and '{sentence_2}': {score}")

model_1 = "sentence-transformers/all-MiniLM-L6-v2"
model_2 = "sentence-transformers/all-mpnet-base-v2"
model_3 = "sentence-transformers/paraphrase-MiniLM-L12-v2"
model_4 = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
model_5 = "sentence-transformers/nli-mpnet-base-v2"

models = [model_1, model_2, model_3, model_4, model_5]


sentence_1 = "rivers woods and hills"
sentence_2 = "streams forests and mountains"
sentence_3 = "deserts sand and shrubs"

for x in models:
    print(f"Now using model: '{x}'")
    print(compare_sentences(sentence_1=sentence_1, sentence_2=sentence_2, model_name=x))
    print(compare_sentences(sentence_1=sentence_1, sentence_2=sentence_3, model_name=x))