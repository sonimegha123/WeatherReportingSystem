# Databricks notebook source
# MAGIC %md
# MAGIC #Installation

# COMMAND ----------

!pip install llama-index-embeddings-azure-openai

# COMMAND ----------

!pip install llama-index-llms-azure-openai

# COMMAND ----------

!pip install llama-index

# COMMAND ----------

!pip install python-dotenv

# COMMAND ----------

!pip install -U -q pinecone-client transformers

# COMMAND ----------

!pip install pinecone-client

# COMMAND ----------

!pip install llama-index-vector-stores-pinecone

# COMMAND ----------

!pip install -U -q sqlalchemy

# COMMAND ----------

!pip install llama-index-callbacks-arize-phoenix

# COMMAND ----------

!pip install pyvis

# COMMAND ----------

!pip install llama-index-readers-wikipedia

# COMMAND ----------

!pip install wikipedia

# COMMAND ----------

!pip install pydantic==1.10.10

# COMMAND ----------

!pip install langchain

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #Imports

# COMMAND ----------

import nest_asyncio

nest_asyncio.apply()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# COMMAND ----------

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# COMMAND ----------

from llama_index.core import ServiceContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.node_parser import TokenTextSplitter
from dotenv import load_dotenv
from dotenv import dotenv_values  
import os 

# COMMAND ----------

# MAGIC %md
# MAGIC #Environment Setup

# COMMAND ----------

env_name = "BuienradarGPT"
config = dotenv_values(env_name)


openai_api_type = "azure"
openai_api_base = "https://test-ds-chatgpt.openai.azure.com/"
openai_api_version = "2023-03-15-preview"
openai_api_key = os.getenv("OPENAI_API_KEY")

# COMMAND ----------

llm = AzureOpenAI(
    model="gpt-35-turbo",
    deployment_name="gpt-35-turbo",
    api_key=openai_api_key,
    azure_endpoint=openai_api_base,
    api_version=openai_api_version,
    temperature=0.0,
    streaming=True,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding",
    api_key=openai_api_key,
    azure_endpoint=openai_api_base,
    api_version=openai_api_version,
)

from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model

# COMMAND ----------

# configurations for vector store
chunk_size = 2048
text_splitter = TokenTextSplitter(chunk_size=2048)
node_parser = SimpleNodeParser()

# COMMAND ----------

# MAGIC %md
# MAGIC #Pinecone setup

# COMMAND ----------

import os
from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key="11166e05-e523-4a61-aa91-bb1e55cdfb28")
pinecone_index = pc.Index("quickstart-index")

# added
pc.describe_index('quickstart-index')

# COMMAND ----------

# MAGIC %md
# MAGIC Set up LLamaIndex PineconeVectorStore!

# COMMAND ----------


from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext


# pinecone namespace
namespace="weather-info"

# connect and name the PineconeVectorStore
locations_vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, namespace=namespace
)

# allow the PineconeVectorStore to be used as storage
storage_context = StorageContext.from_defaults(vector_store=locations_vector_store)

# allow the creation of an Index
weather_vector_index = VectorStoreIndex([], storage_context=storage_context)

# COMMAND ----------

# MAGIC %md
# MAGIC #Data Loading & Pre-processing

# COMMAND ----------

import pandas as pd

weather_df = pd.read_csv("/dbfs/mnt/ds-data-apps/megha/weather_data/weather_reports.csv",  sep=';')

# COMMAND ----------

weather_df['publishDate'] = pd.to_datetime(weather_df['publishDate'])

# Filter rows where the Date is from the year 2023
filtered_df = weather_df[weather_df['publishDate'].dt.year == 2023]
filtered_df['date'] = filtered_df['publishDate'].dt.date
filtered_df['time'] = filtered_df['publishDate'].dt.strftime('%H:%M:%S')  # Convert time to string format

df = filtered_df[['date', 'time', 'publishDate', 'content']]
df = df.rename(columns={'content': 'weather_report'})

df.drop(columns=['publishDate'], inplace=True)


# COMMAND ----------

from bs4 import BeautifulSoup

# converting HTML to plain text
df['weather_report'] = df['weather_report'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())

display(df)

# COMMAND ----------

df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')


# function to get season based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

# Add 'season' column based on 'date' column
df['month'] = pd.to_datetime(df['date']).dt.month
df['season'] = df['month'].apply(get_season)

# Drop the temporary 'month' column if you don't need it
df.drop(columns=['month'], inplace=True)

display(df)

# COMMAND ----------

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)

engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
metadata_obj = MetaData()

# COMMAND ----------

import pandas as pd
from sqlalchemy.engine.base import Engine

def add_df_to_sql_database(table_name: str, pandas_df: pd.DataFrame, engine: Engine) -> None:
  pandas_df.to_sql(table_name, engine)


add_df_to_sql_database("weather_data", df, engine)
table_names = ["weather_data"]

# COMMAND ----------

from sqlalchemy import text

with engine.connect() as conn:
  result = conn.execute(text("SELECT weather_report FROM weather_data WHERE date = '2023-05-24'"))
  for row in result:
    print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC Great! Now that we've done that - let's add some context by grabbing some information about the Netherland's weather from Wikipedia!

# COMMAND ----------

!pip install -U -q wikipedia

# COMMAND ----------

from llama_index.readers.wikipedia import WikipediaReader
import wikipedia as wp

season_list = []
with engine.connect() as conn:
  results = conn.execute(text("SELECT DISTINCT season FROM weather_data"))
  seasons = [result.season for result in results]

  for season in seasons:
        # search_results = wp.search(season + " in Netherlands weather", results=100)
        search_results = wp.search(season + " weather Netherlands", results=100)
        filtered_results = [result for result in search_results if "Netherlands" in result or 'Holland' in result]
        season_list.extend(filtered_results)

print(season_list)




# COMMAND ----------

from llama_index.readers.wikipedia import WikipediaReader

wiki_docs = WikipediaReader().load_data(pages=season_list, auto_suggest=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Index Creation  
# MAGIC SQL Index

# COMMAND ----------

from llama_index.core import SQLDatabase

sql_database = SQLDatabase(engine, include_tables=table_names)


# COMMAND ----------

# MAGIC %md
# MAGIC SQL-to-text  
# MAGIC Now we can set up our NLSQLTableQueryEngine.
# MAGIC Converts natural language queries to SQL queries and queries against the database.

# COMMAND ----------

from llama_index.core.indices.struct_store.sql_query import NLSQLTableQueryEngine

sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=table_names, llm=llm
)

# COMMAND ----------

# MAGIC %md
# MAGIC vector index

# COMMAND ----------

from llama_index.core import Settings

for season, wiki_doc in zip(season_list, wiki_docs):
    try:
        nodes = node_parser.get_nodes_from_documents([wiki_doc])
        for node in nodes:
            node.metadata = {"title": season}
        weather_vector_index.insert_nodes(nodes)
    except Exception as e:
        print(f"Error processing season {season}: {e}")
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC #Setting up query engines

# COMMAND ----------

from llama_index.core.query_engine import SQLAutoVectorQueryEngine, RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo

weather_vector_store_info = VectorStoreInfo(
    content_info="Articles about different seasons and activities in the Netherlands",
    metadata_info=[
        MetadataInfo(
            name="title",
            type="str",
            description="Season of the Netherland")
    ]
)

weather_vector_auto_retriever = VectorIndexAutoRetriever(
    weather_vector_index, vector_store_info=weather_vector_store_info
)

weather_retriever_query_engine = RetrieverQueryEngine.from_args(
    weather_vector_auto_retriever, llm=llm
)

# COMMAND ----------

from llama_index.core.query_engine import SQLAutoVectorQueryEngine

sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    name="sql_tool",
    description=(
        "The Weather Report Query Tool translates natural language queries into SQL queries to retrieve weather data from the 'weather_data' table. "
        "This table includes information on date, time, weather_report, and season, with dates formatted as 'YYYY-MM-DD'. "
        "The tool is designed to provide insights on historical weather conditions in the Netherlands based on the user's specified date. "
        "It not only fetches the relevant weather report but also analyzes and interprets the data to answer the user's questions effectively. "
        "Use this tool to understand what the weather was like on a particular day, aiding in various weather-related inquiries and analyses."
    ),   
)


vector_tool = QueryEngineTool.from_defaults(
    query_engine=weather_retriever_query_engine,
    name="vector_tool",
    description=(
        "The General Weather Query Tool is designed to answer broad questions about the weather in the Netherlands when no specific date is provided in the user's query. "
        "Using advanced retrieval techniques, this tool can provide comprehensive information and insights about typical weather patterns, seasonal variations, and general climate characteristics of the Netherlands. "
        "Ideal for users seeking an understanding of Dutch weather without the need for date-specific data."
    ),
)


from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector

# query_engine = RouterQueryEngine(
#     selector=LLMSingleSelector.from_defaults(),
#     query_engine_tools=([sql_tool] + [vector_tool]),
# )

query_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=([sql_tool] + [vector_tool]),
)

# COMMAND ----------

response = query_engine.query("Is it a good day to for a barbecue on July 10, 2023?")
print(str(response))

response = query_engine.query("what kind of clothes should I pack for my trip in mid November in the Netherlands?")
print(str(response))

# COMMAND ----------

# MAGIC %md
# MAGIC #_Chatbot_ _Agent_

# COMMAND ----------

while True:
        question = input("> ")
        if question == "exit":
            break
        response = query_engine.query(question) 
        print(str(response))

# COMMAND ----------


import time

session_state = {
    "messages": [
        {"role": "assistant", "content": "Ask me a question!"}
    ]
}

def queryDB(prompt):
  response = query_engine.query(prompt)
  return response

def pprint_response(response, show_source=True):
    print(response)


def run_chatbot():
    while True:
        prompt = input("Your question (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            print("Exiting the chatbot. Goodbye!")
            break

        if prompt:
            session_state["messages"].append({"role": "user", "content": prompt})

        # Generating and displaying the response
        if session_state["messages"][-1]["role"] != "assistant":
            print("Assistant: Thinking...")
            time.sleep(2)
            response = queryDB(prompt)
            message = {"role": "assistant", "content": response}
            session_state["messages"].append(message)
            
            pprint_response(response, show_source=True)


run_chatbot()

# COMMAND ----------

# MAGIC %md
# MAGIC #More Inferencing

# COMMAND ----------

import json

query_str = f'Tell me about the winters in the Netherlands?'
response = query_engine.query(query_str)

print(str(response))

# COMMAND ----------


query_str = f'plan a 2-day itenerary for me suggesting some fun activities for fall season?'
response = query_engine.query(query_str)
print(str(response))

# COMMAND ----------

date = "January 6, 2023"
query_str = f'can you plan a 2 day itenerary for me suggesting some fun activities for summers?'
response = query_engine.query(query_str)
print(str(response))

# COMMAND ----------

query_str = "What does the weather look like on June 1st, 2023?"
response = query_engine.query(query_str)
print(str(response))

# COMMAND ----------

query_str = "Is June 23rd, 2023 a good day for a barbecue"
response = query_engine.query(query_str)
print(str(response))
