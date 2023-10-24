# Atlas Vector Search with RAG

The Python scripts in this repo use Atlas Vector Search with Retrieval-Augmented Generation (RAG) architecture to build a Question Answering application. They use the LangChain framework, OpenAI models, as well as Gradio in conjunction with Atlas Vector Search in a RAG architecture, to create this app.


## Setting up the Environment

1. Install the following packages:
```
pip3 install langchain pymongo bs4 openai tiktoken gradio requests lxml argparse unstructured
```
2. Create OpenAI API Key from [here](https://platform.openai.com/account/api-keys). Note that this requires a paid account with OpenAI, with enough credits. OpenAI API requests stop working if credit balance reaches `$0`.

3. Save the OpenAI API key and the MongoDB URI in the `key_param.py` file, like this:
```
openai_api_key = "ENTER_OPENAI_API_KEY_HERE"
MONGO_URI = "ENTER_MONGODB_URI_HERE"
```
4. Use the following two python scripts:
   - **load_data.py**: This script will be used to load your documents and ingest the text and vector embeddings, in a MongoDB collection.
   - **extract_information.py**: This script will generate the user interface and will allow you to perform question-answering against your data, using Atlas Vector Search and OpenAI.

**Note:** In this demo, I've used:
   - DB Name: `langchain_demo`
   - Collection Name: `collection_of_text_blobs`
   - The text files that I am using as my source data are saved in a directory named `sample_files`.
