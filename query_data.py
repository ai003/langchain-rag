import argparse
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from openai import OpenAI
from dotenv import load_dotenv
import os
from get_embedding_function import get_embedding_function


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

print(client)  # test out object

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = '''
Anser the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
 
'''


def main():
    print("enter main")
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text")
    args = parser.parse_args()
    query_text = args.query_text
    print("right before run called")
    query_rag(query_text)
    print("after")


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=embedding_function)
    # retrieve relevant context
    # list top k most relevant chunks to question
    results = db.similarity_search_with_score(query_text, k=5)

    # use this with original question text to generate the prompt
    context_text = "\n\n --- \n\n".join(
        doc.page_content for doc, _score in results)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    print("openai being called")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    response_text = response.choices[0].message.content.strip()
    print("after", response_text)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
