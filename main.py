import os
import yaml
import logging
import google.cloud.logging
from flask import Flask, render_template, request

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from langchain_google_vertexai import VertexAIEmbeddings

# Configure Cloud Logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# Read application variables from the config fle
BOTNAME = "FreshBot"
SUBTITLE = "Your Friendly Restaurant Safety Expert"

app = Flask(__name__)

# Initializing the Firebase client
db = firestore.Client()

# TODO: Instantiate a collection reference
collection = db.collection("food-safety")

# TODO: Instantiate an embedding model here
embedding_model = VertexAIEmbeddings(model_name="text-embedding-005")

# TODO: Instantiate a Generative AI model here
gen_model = GenerativeModel("gemini-2.0-flash-001")
# gen_model = GenerativeModel("gemini-2.5-pro-preview-06-05")


# TODO: Implement this function to return relevant context
# from your vector database
# def search_vector_database(query: str):

#     context = ""

#     # 1. Generate the embedding of the query

#     # 2. Get the 5 nearest neighbors from your collection.
#     # Call the get() method on the result of your call to
#     # find_neighbors to retrieve document snapshots.

#     # 3. Call to_dict() on each snapshot to load its data.
#     # Combine the snapshots into a single string named context


#     # Don't delete this logging statement.
#     logging.info(
#         context, extra={"labels": {"service": "cymbal-service", "component": "context"}}
#     )
#     return context

def search_vector_database(query: str):
    context = ""

    # Step 1: Embed the user query
    query_embedding = embedding_model.embed_query(query)

    # Step 2: Find 5 most similar chunks using vector search
    # Call the get() method on the result of your call to
    # find_neighbors to retrieve document snapshots.
    docs = (
        collection.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_embedding),
            distance_measure=DistanceMeasure.DOT_PRODUCT,
            limit=5,
        ).get()
    )

    # Step 3: Call to_dict() on each snapshot to load its data.
    # Concatenate content from top documents
    context = "\n\n".join([doc.to_dict()["content"] for doc in docs])

    # Don't delete this logging statement.
    logging.info(
        context, extra={"labels": {"service": "cymbal-service", "component": "context"}}
    )

    return context

# TODO: Implement this function to pass Gemini the context data,
# generate a response, and return the response text.
# def ask_gemini(question):

#     # 1. Create a prompt_template with instructions to the model
#     # to use provided context info to answer the question.
#     prompt_template = ""

#     # 2. Use your search_vector_database function to retrieve context
#     # relevant to the question.
    
#     # 3. Format the prompt template with the question & context

#     # 4. Pass the complete prompt template to gemini and get the text
#     # of its response to return below.
#     response = "Not implemented."

#     return response

def ask_gemini(question):
    # Step 1: Define a structured prompt
    prompt_template = """
    You are a helpful assistant answering questions based on food safety training material.

    Context:
    {context}

    Question:
    {question}

    Answer:"""

    # Step 2: Get relevant context using vector search
    context = search_vector_database(question)

    # Step 3: Fill in the prompt
    prompt = prompt_template.format(context=context, question=question)

    # Step 4: Get Gemini response
    response = gen_model.generate_content(prompt).text

    return response

# The Home page route
@app.route("/", methods=["POST", "GET"])
def main():

    # The user clicked on a link to the Home page
    # They haven't yet submitted the form
    if request.method == "GET":
        question = ""
        answer = "Hi, I'm FreshBot, what can I do for you?"

    # The user asked a question and submitted the form
    # The request.method would equal 'POST'
    else:
        question = request.form["input"]
        # Do not delete this logging statement.
        logging.info(
            question,
            extra={"labels": {"service": "cymbal-service", "component": "question"}},
        )
        
        # Ask Gemini to answer the question using the data
        # from the database
        answer = ask_gemini(question)

    # Do not delete this logging statement.
    logging.info(
        answer, extra={"labels": {"service": "cymbal-service", "component": "answer"}}
    )
    print("Answer: " + answer)

    # Display the home page with the required variables set
    config = {
        "title": BOTNAME,
        "subtitle": SUBTITLE,
        "botname": BOTNAME,
        "message": answer,
        "input": question,
    }

    return render_template("index.html", config=config)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
