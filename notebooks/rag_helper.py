import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# Building a prompt with the search results

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""


class RAGHelper:
    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        course="llm-zoomcamp",
        prompt_template=PROMPT_TEMPLATE,
        model="claude-haiku-4-5-20251001",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, question: str, num_results: int = 5) -> list:
        """
        Search for relevant documents based on the question and course.
        """
        return self.index.search(
            question,
            boost_dict={"question": 2},
            filter_dict={"course": self.course},
            num_results=num_results,
        )

    # Function to build the context
    def build_context(self, search_results: list) -> str:
        context = []
        for doc in search_results:
            context.append(doc["section"])
            context.append("Q: " + doc["question"])
            context.append("A: " + doc["answer"])
            context.append("")  # Add a blank line for better readability
        return "\n".join(context)

    def build_prompt(self, question: str, search_results: list) -> str:
        context = self.build_context(search_results)
        prompt = self.prompt_template.format(question=question, context=context)
        return prompt.strip()

    def llm(self, prompt: str) -> str:
        """
        Simple function to call the LLM and return the response with Anthropic API.
        """
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=300,
            system=self.instructions,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )

    def rag(self, query: str) -> str:
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer
