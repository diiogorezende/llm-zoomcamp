import faiss
import numpy as np
import requests


def load_faq_data():
    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_raw:
        course_url = f"{url_prefix}/{course['path']}"
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(course_data)
    return documents


def build_faiss_index(documents, model, batch_size=50):
    texts = [doc["question"] + " " + doc["answer"] for doc in documents]
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors.extend(model.encode(batch))

    X = np.array(vectors).astype(np.float32)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    return index
