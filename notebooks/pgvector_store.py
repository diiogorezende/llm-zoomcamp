import numpy as np
import psycopg
from pgvector.psycopg import register_vector


class PGVectorStore:
    def __init__(self, database_url: str, embedder):
        self.embedder = embedder
        self.connection = psycopg.connect(database_url)
        self._create_schema()
        register_vector(self.connection)

    def _create_schema(self):
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS faq_documents (
                id BIGINT PRIMARY KEY,
                course TEXT NOT NULL,
                section TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                embedding vector(384) NOT NULL
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS faq_documents_embedding_idx
                ON faq_documents
                USING hnsw(embedding vector_cosine_ops)
                """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS faq_documents_course_ids
                ON faq_documents (course)
                """)

        self.connection.commit()

    def ingest(self, documents: list[dict]):
        texts = [
            f"{document['question']} {document['answer']}" for document in documents
        ]

        embeddings = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )

        rows = [
            (
                index,
                document["course"],
                document["section"],
                document["question"],
                document["answer"],
                embedding,
            )
            for index, (document, embedding) in enumerate(
                zip(documents, embeddings),
                start=1,
            )
        ]

        with self.connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO faq_documents(
                    id, course, section, question, answer, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    course = EXCLUDED.course,
                    section = EXCLUDED.section,
                    question = EXCLUDED.question,
                    answer = EXCLUDED.answer,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )
        self.connection.commit()

    def search(self, query: str, course: str, num_results: int = 5) -> list[dict]:
        query_embedding = self.embedder.encode(
            query,
            normalize_embeddings=True,
        ).astype(np.float32)

        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, course, section, question, answer
                FROM faq_documents
                WHERE course = %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (course, query_embedding, num_results),
            )
            columns = [column.name for column in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
