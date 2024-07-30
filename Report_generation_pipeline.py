# Databricks notebook source
# MAGIC %md
# MAGIC #Installation

# COMMAND ----------

pip install --upgrade pip

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

!pip install llama-index-embeddings-azure-openai
!pip install llama-index-llms-azure-openai
!pip install openai


# COMMAND ----------

# %pip install llama-index-llms-azure-openai

# COMMAND ----------

!pip install llama-index

# COMMAND ----------

pip install python-dotenv

# COMMAND ----------

!pip install -U -q pinecone-client transformers

# COMMAND ----------

pip install pinecone-client

# COMMAND ----------

!pip install llama-index-vector-stores-pinecone

# COMMAND ----------

# !pip install openai

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
# MAGIC #Environmenet Setup

# COMMAND ----------

env_name = "weather2022_onwards.csv"
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
# MAGIC #Data loading & Pre processing

# COMMAND ----------

!pip install -U -q sqlalchemy

# COMMAND ----------

import pandas as pd

de_bilt_df = pd.read_csv("/dbfs/mnt/ds-data-apps/megha/weather_data/De_Bilt.csv")
Eindhoven_df = pd.read_csv("/dbfs/mnt/ds-data-apps/megha/weather_data/Eindhoven.csv")
Amsterdam_df = pd.read_csv("/dbfs/mnt/ds-data-apps/megha/weather_data/Amsterdam.csv")
Maastricht_df = pd.read_csv("/dbfs/mnt/ds-data-apps/megha/weather_data/Maastricht.csv")
Rotterdam_df = pd.read_csv("/dbfs/mnt/ds-data-apps/megha/weather_data/Rotterdam.csv")

list_of_dfs = [de_bilt_df, Eindhoven_df, Amsterdam_df, Maastricht_df, Rotterdam_df]

# COMMAND ----------

# Function to convert wind speed in Beaufort scale values
def beaufort_scale(fh_value):
    if fh_value < 1:
        return 0
    elif 1 <= fh_value <6:
        return 1
    elif 6 <= fh_value <12:
        return 2
    elif 12 <= fh_value <20:
        return 3
    elif 20 <= fh_value <29:
        return 4
    elif 29 <= fh_value <38:
        return 5
    elif 38 <= fh_value <50:
        return 6
    elif 50 <= fh_value <62:
        return 7
    elif 62 <= fh_value <75:
        return 8
    elif 75 <= fh_value <89:
        return 9
    elif 89 <= fh_value <103:
        return 10
    elif 103 <= fh_value <118:
        return 11
    else:
        return 12
      


# COMMAND ----------

# Function to convert wind direction in degrees to actual direction
def wind_direction(degrees):
    if degrees == 0:
        return 'calm'
    elif degrees == 990:
        return 'variable'
    elif 0 < degrees < 90:
        return 'NE'
    elif degrees == 90:
        return 'E'
    elif 90 < degrees < 180:
        return 'SE'
    elif degrees == 180:
        return 'S'
    elif 180 < degrees < 270:
        return 'SW'
    elif degrees == 270:
        return 'W'
    elif 270 < degrees < 360:
        return 'NW'
    elif degrees == 360:
        return 'N'
    else:
        return 'Invalid direction'





# COMMAND ----------

# Function to categorize hours
def categorize_hour(hour):
    if 5 <= hour <= 11:
        return 'Morning'
    elif 12 <= hour <= 17:
        return 'Afternoon'
    elif 18 <= hour <= 24:
        return 'Evening'
    else:  # Hour 1 to 4
        return 'Night'

# COMMAND ----------

avg_temperatures = {
    1: 3.45,
    2: 3.9,
    3: 6.45,
    4: 9.3,
    5: 12.95,
    6: 15.6,
    7: 17.9,
    8: 17.7,
    9: 15.5,
    10: 11.55,
    11: 7.3,
    12: 4.2
}

def add_avg_temperature_column(df):
    df['Date'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df['Month'] = df['Date'].dt.month
    df['Avg_Temperature'] = df['Month'].map(avg_temperatures)

# COMMAND ----------

import math

for df in list_of_dfs:
    # Rename columns by removing extra spaces
    new_columns_df = {col: col.strip() for col in df.columns}
    df.rename(columns=new_columns_df, inplace=True)
    
    # convert the temperature to 1 degree Celcius
    df[['T', 'T10N', 'TD']] = df[['T', 'T10N', 'TD']].apply(pd.to_numeric, errors='coerce')
    df[['T', 'T10N', 'TD']] *= 0.1

    df['FH'] = df['FH'] * 0.1 * (18/5)  # convert wind speed in km/h
    df['FH'] = df['FH'].apply(beaufort_scale)

    df['FX'] = df['FX'] *0.1 * (18/5)  # convert wind gust in km/h

    df['DD'] = df['DD'].apply(wind_direction) # convert wind degrees to direction

    add_avg_temperature_column(df)

    df['T'] = df['T'].apply(lambda x: math.ceil(x))

    df['Time_Category'] = df['HH'].apply(categorize_hour)

    


# COMMAND ----------

req_cols = ['STN', 'YYYYMMDD', 'Time_Category', 'T', 'HH', 'DD', 'FH', 'FX', 'SQ', 'DR', 'RH', 'VV', 'N', 'M', 'R', 'S', 'O', 'Y', 'Avg_Temperature']

new_column_names = ['station', 'date', 'Time_Category', 'temperature', 'hour', 'winddirection', 'beaufort', 'wind_gust', 'sunshine', 'precipitationduration', 'precipitation', 'visibility', 'cloudcover', 'fog', 'rain', 'snow', 'thunderstorm', 'iceformation', 'Avg_Temperature']

de_bilt_df = de_bilt_df[req_cols]
Eindhoven_df = Eindhoven_df[req_cols]
Amsterdam_df = Amsterdam_df[req_cols]
Maastricht_df = Maastricht_df[req_cols]
Rotterdam_df = Rotterdam_df[req_cols]

de_bilt_df = de_bilt_df.round(1)
Eindhoven_df = Eindhoven_df.round(1)
Amsterdam_df = Amsterdam_df.round(1)
Maastricht_df = Maastricht_df.round(1)
Rotterdam_df = Rotterdam_df.round(1)

de_bilt_df.rename(columns=dict(zip(de_bilt_df.columns, new_column_names)), inplace=True)
Eindhoven_df.rename(columns=dict(zip(Eindhoven_df.columns, new_column_names)), inplace=True)
Amsterdam_df.rename(columns=dict(zip(Amsterdam_df.columns, new_column_names)), inplace=True)
Maastricht_df.rename(columns=dict(zip(Maastricht_df.columns, new_column_names)), inplace=True)
Rotterdam_df.rename(columns=dict(zip(Rotterdam_df.columns, new_column_names)), inplace=True)


# COMMAND ----------

display(Rotterdam_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have the dataframe, let's create the SQL Table!

# COMMAND ----------

# Maastricht_df.to_csv('/dbfs/mnt/ds-data-apps/megha/AnnotationLab_data/Maastricht.csv', index=False)

# COMMAND ----------

from sqlalchemy import create_engine

engine = create_engine("sqlite+pysqlite:///:memory:", future=True)

# COMMAND ----------

type(engine)

# COMMAND ----------

import pandas as pd
from sqlalchemy.engine.base import Engine

def add_df_to_sql_database(table_name: str, pandas_df: pd.DataFrame, engine: Engine) -> None:
  pandas_df.to_sql(table_name, engine)

# COMMAND ----------

add_df_to_sql_database("De_Bilt", de_bilt_df, engine)
add_df_to_sql_database("eindhoven", Eindhoven_df, engine)
add_df_to_sql_database("Rotterdam", Rotterdam_df, engine)
add_df_to_sql_database("Amsterdam", Amsterdam_df, engine)
add_df_to_sql_database("Maastricht", Maastricht_df, engine)

# COMMAND ----------

table_names = ["De_Bilt", "eindhoven", "Rotterdam", "Amsterdam", "Maastricht"]

# COMMAND ----------

# MAGIC %md
# MAGIC #SQL query response check

# COMMAND ----------

from sqlalchemy import text

with engine.connect() as conn:
  # result = conn.execute(text("SELECT YYYYMMDD, T FROM De_Bilt WHERE YYYYMMDD LIKE '202301%' AND T = (SELECT MAX(T) FROM De_Bilt WHERE YYYYMMDD LIKE '202301%')\nORDER BY T DESC LIMIT 1;"))
  result = conn.execute(text("""SELECT Time_Category, temperature, winddirection, beaufort, precipitation, sunshine, cloudcover, fog, rain, snow, thunderstorm FROM Amsterdam 
WHERE date = 20230510 AND (hour BETWEEN 5 AND 17 OR hour BETWEEN 18 AND 23 OR hour BETWEEN 0 AND 4)
UNION
SELECT Time_Category, temperature, winddirection, beaufort, precipitation, sunshine, cloudcover, fog, rain, snow, thunderstorm 
FROM Amsterdam 
WHERE date = 20230511
ORDER BY Time_Category ASC"""))
  for row in result:
    print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC #SQL Index

# COMMAND ----------

from llama_index.core import SQLDatabase

sql_database = SQLDatabase(engine, include_tables=table_names)

# COMMAND ----------

# MAGIC %md
# MAGIC text-to-sql

# COMMAND ----------

from llama_index.core.indices.struct_store.sql_query import NLSQLTableQueryEngine

query_engine = NLSQLTableQueryEngine(sql_database)

# COMMAND ----------

response = query_engine.query("Select all the data from first of january, 2023 in Amsterdam")


# COMMAND ----------

# MAGIC %md
# MAGIC Now we can set up our NLSQLTableQueryEngine.  
# MAGIC Converts natural language queries to SQL queries and queries against the database.

# COMMAND ----------

from llama_index.core.indices.struct_store.sql_query import NLSQLTableQueryEngine

sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=table_names, llm=llm
)



# COMMAND ----------

query_str = "Can you generate the weather report for date 20230913 in Rotterdam? Write the report in a funny style and it can be upto 5 sentences long."
response = sql_query_engine.query(query_str)
response
# context_weather_data = response.source_nodes[0].text


# COMMAND ----------

# Refer to this when building the chatbot
query_str = "Today is 20230410. Can I wear short trousers and a t-shirt outside? Answer with yes or no"
response = sql_query_engine.query(query_str)
response

# COMMAND ----------

# MAGIC %md
# MAGIC #Response in a desired way

# COMMAND ----------

# MAGIC %md
# MAGIC ##English

# COMMAND ----------

from IPython.display import Markdown, display

response_template = """
## Question
> {question}

## Answer
{response}

## Generated SQL Query
```
{sql}
```
"""


def chat_to_sql(
    question: str | list[str],
    tables: list[str] | None = None,
    # format_hint: str = "Please generate a weather report that gives 3 lines about overall weather of the day. And one short paragraph for morning, afternoon, and evening each",
    format_hint: str = """  
Generate a weather report in for the given location and day in the Netherlands. Provide a comprehensive overview of the weather conditions for the specified day and the following day. Ensure that the report is written in a professinal tone, resembling that of a meteorologist delivering the forecast with warmth and approachability. Keep the discussion of today's weather to around 3 paragraphs of around 3 sentences each. Additionally, include a paragraph of around 3 sentences to offer a brief glimpse into tomorrow's weather. The entire report should be of atleast 8 sentences in length. Include variations while writing the report to keep it interesting. 

For the location and day, analyze the weather data to create a detailed report. Begin by summarizing the overall weather forecast for both days in a paragraph that paints a vivid picture of what to expect. This should be followed by a paragraph for the morning, starting with "This morning", where you can bring to life the atmospheric conditions with poetic descriptions. Subsequently, provide a paragraph for the afternoon, starting with "This afternoon", continuing the friendly tone and professional poetic language. Then, describe the evening and night with a paragraph starting with "Tonight", capturing the essence of the nighttime weather in an engaging manner. Finally, offer a few lines about the next day, starting with "Tomorrow" or "The weekend" in case of a weekend, keeping the tone light and the descriptions lively. Do not divide it in morning, afternoon and different sections, instead give an overall weather forcast of tomorrow. If there is significant weather approaching/passing in the evening, we mention it in this section as well. For example if it was dry the whole day and there is heavy showers passing in the evening, then one would add a sentence like "Sunny and dry, but with some showers in the evening.

Weather Report Format:
Please consider the following hour for each section of the report:
When talking about today, look at the data only for the given date.
- When talking about today morning, look at the rows with hour values 5-11 for the given date. 
- When talking about today afternoon, look at the rows with hour values 12-17 for the given date. 
- When talking about today Evening adn night, look at the rows with hour values 18-4 for the given date.
When writing about tomorrow, look at the data only for the next date.
- When talking about tomorrow, look at all rows for the next day.

Follow the provided example format for the weather report:

Start with a general overview of the weather for today.
Provide specific details for each time period, including weather, precipitation, temperature, and wind conditions.
Use descriptive language to convey the atmospheric conditions, such as "sunny afternoon", "cloudy with light rain", or "partly cloudy with a gentle breeze". 
Give a general overview of tomorrow's weather relatively.
Ensure that the report is written in a friendly tone, resembling that of a meteorologist delivering the forecast with warmth and approachability. Be creative and include variations while writing the report to keep it interesting. 

Clouds and Precipitation:
The report should not contain values of cloudcover or precipitation, instead only use them to describe the weather condition. Consider the following parameters to assess cloud cover and precipitation and provide descriptive weather descriptions in the report accordingly:
- SQ (Duration of sunshine per hourly period): Use SQ values to determine the presence of sunshine and its duration.
- N (Cloud cover): Assess cloud cover based on N values(9=upper sky invisible) and provide descriptive weather descriptions accordingly.
- R (Rainfall): Determine the occurrence of rainfall based on R values.
- RH (Relative Humidity (Hourly precipitation amount (in 0.1 mm) (-1 for <0.05 mm) / Hourly precipitation amount (in 0.1 mm) (-1 for <0.05 mm)))
- O (Thunderstorm (0=did not occur; 1=occurred in the previous hour))
- S (Snowfall (0=did not occur; 1=occurred in the previous hour))
- M (Mist (0=did not occur; 1=occurred in the previous hour))
- VV (Horizontal visibility (Horizontal visibility during observation (0=less than 100m; 1=100-200m; 2=200-300m;...; 49=4900-5000m; 50=5-6km; 56=6-7km; 57 =7-8km; ...; 79=29-30km; 80=30-35km; 81=35-40km;...; 89=more than 70km) / Horizontal visibility at the time of observation (0=less than 100m; 1=100-200m; 2=200-300m;...; 49=4900-5000m; 50=5-6km; 56=6-7km; 57=7-8km; ...; 79=29- 30km; 80=30-35km; 81=35-40km;...; 89=more than 70km))

Based on the parameters, describe the weather conditions as follows:
Look at values of sunshine, cloudcover, rain, snow, fog, visibility, thunderstorm, fog values to describe the weather conditions as follows:
- if sunshine values indicate ample sunshine, describe as "Sunny afternoon" or "Sunny day".
- for high cloudcover and low sunshine values describe as "Gray day".
- if fog equals 1 and visibility indicates how much fog is there describe as "Foggy".
- for moderate sunshine, high cloudcover, and rain equals 1, describe as "Sun and showers".
- if thunderstorm equals 1, rain equals 1, and sunshine is non 0 with some cloudcover, describe as "Thunderstorms".
- for non 0 sunshine and some cloudcover, describe as "Sun and cumulus clouds".
- if rain equals 1, various precipitation values, and sunshine is low or 0, describe as "Occasional rain".
- if fog equals 1 and sunshine increases with time describe as "Fog followed by sun".
- for rain equals 1, describe as "Rain".
- for high cloudcover and low or no sunshine describe as "Gray and hardly sunny".
- for some sunshine, thunderstorm equals 1, and some cloudcover, describe as "Cumulus clouds and thunderstorms".
- if snow equals 1, and various precipitation values, describe as "Snow showers".
- for some sunshine, snow equals 1, and various precipitation values describe as "Sun and snow showers".
- if low or 0 sunshine, rain equals 1, snow equals 1, and various precipitation values describe as "Winter showers".

Temperature (temperature in celsius):
Discuss the temperature variations throughout the day, analyzing data from the "temperature" column. Pick the write temperature values for both the dates. 

Wind:
Describe the wind conditions using the Beaufort wind force scale, taking into account the following features:
- DD (Wind Direction)
- FH (Beaufort wind force)
Discuss the wind force descriptions based on Beaufort scale values (FH):
0: Calm, 1: Light Air, 2: Light Breeze, 3: Gentle Breeze, 4: Moderate Breeze, 5: Fresh Breeze, 6: Strong Breeze, 7: Near Gale, 8: Gale, 9: Strong Gale, 10: Storm, 11: Violent Storm, 12: Hurricane.
Ensure to describe wind conditions in a descriptive and vivid manner, utilizing the Beaufort scale descriptions to convey the strength and impact of the wind.

Table Selection:
Ensure to utilize the correct table named after the location requested by the user to fetch the weather data. If the location is Eindhoven, use the table with station value 370. If the location is De Bilt, use the table with station value 260. If the location is Amsterdam, use the table with station value 240. If the location is Maastricht, use the table with station value 380. And if the location is Rotterdam, use the table with station value 344. The dates are in the format YYYYMMDD in the tables. 
To provide a comprehensive weather report, begin by retrieving the data for the current day to discuss today's weather conditions. 
Subsequently, ensure to also retrieve the data for the following day to include tomorrow's weather forecast in the report. 



    """


    ,
    synthesize_response: bool = True,
):
    
    tables = tables 
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=tables,
        synthesize_response=synthesize_response,
    )
    query_str = question + "\n" + format_hint
    try:
        response = query_engine.query(query_str)
        response_md = str(response)
        sql = response.metadata["sql_query"]
    except Exception as ex:
        response_md = "`⚠️ No data ⚠️`"
        sql = f"ERROR: {str(ex)}"


    display(Markdown(response_template.format(
        question=question,
        response=response_md,
        sql=sql,
    )))

# call the function here
chat_to_sql("location - Amsterdam, date - May 10, 2023")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Dutch

# COMMAND ----------

from IPython.display import Markdown, display

response_template = """
## Question
> {question}

## Answer
{response}

## Generated SQL Query
```
{sql}
```

"""


def chat_to_sql_dutch(
    question: str | list[str],
    tables: list[str] | None = None,
    # format_hint: str = "Please generate a weather report that gives 3 lines about overall weather of the day. And one short paragraph for morning, afternoon, and evening each",
    format_hint: str = """  
Genereer een weerbericht in de Nederlandse taal voor de opgegeven locatie en dag in Nederland. Geef een uitgebreid overzicht van de weersomstandigheden voor de aangegeven dag en de volgende dag. Zorg ervoor dat het rapport op een professionele toon is geschreven, die lijkt op die van een meteoroloog die de voorspelling op een warme en laagdrempelige manier levert. Beperk de bespreking van het weer van vandaag tot ongeveer 3 paragrafen van elk ongeveer 3 zinnen. Voeg daarnaast een alinea van ongeveer drie zinnen toe om een ​​korte blik te werpen op het weer van morgen. Het volledige rapport moet minimaal 8 zinnen lang zijn. Voeg tijdens het schrijven van het rapport variaties toe om het interessant te houden. 

Analyseer voor de locatie en dag de weergegevens om een ​​gedetailleerd rapport te maken. Begin met het samenvatten van de algemene weersvoorspelling voor beide dagen in een paragraaf die een levendig beeld schetst van wat u kunt verwachten. Dit moet worden gevolgd door een paragraaf voor de ochtend, beginnend met "Vanmorgen", waarin je de atmosferische omstandigheden tot leven kunt brengen met poëtische beschrijvingen. Geef vervolgens een paragraaf voor de middag, beginnend met "Vanmiddag", waarbij de vriendelijke toon en het professionele poëtische taalgebruik worden voortgezet. Beschrijf vervolgens de avond en de nacht met een paragraaf die begint met 'Vanavond', waarin u de essentie van het nachtelijke weer op een boeiende manier vastlegt. Bied ten slotte een paar regels aan over de volgende dag, te beginnen met 'Morgen' of 'Het weekend' als het een weekend is, waarbij u de toon licht en de beschrijvingen levendig houdt. Verdeel het niet in ochtend, middag en verschillende secties, maar geef in plaats daarvan een algemeen weerbericht voor morgen. Als er 's avonds significant weer nadert/overgaat, vermelden we dit ook in deze sectie. Als het bijvoorbeeld de hele dag droog is geweest en er 's avonds zware buien vallen, dan zou je een zin toevoegen als 'Zonnig en droog, maar 's avonds met enkele buien.

Structuur weerrapport:
Formaat weerrapport:
Houd voor elk onderdeel van het rapport rekening met het volgende uur:
Als je het over vandaag hebt, kijk dan alleen naar de gegevens voor de gegeven datum.
- Als het over vandaagochtend gaat, kijk dan naar de rijen met uurwaarden 5-11 voor de opgegeven datum.
- Als het over vandaagmiddag gaat, kijk dan naar de rijen met uurwaarden 12-17 voor de opgegeven datum.
- Als het over vandaag, avond en nacht gaat, kijk dan naar de rijen met uurwaarden 18-4 voor de gegeven datum.
Wanneer u over morgen schrijft, kijk dan alleen naar de gegevens voor de volgende datum.
- Als je het over morgen hebt, kijk dan naar alle rijen voor de volgende dag.

Volg het meegeleverde voorbeeldstructuur voor het weerbericht:

Begin met een algemeen overzicht van het weer voor vandaag.
Geef specifieke details op voor elke tijdsperiode, inclusief weer, neerslag, temperatuur (in afgeronde waarden) en windomstandigheden.
Gebruik beschrijvende taal om de atmosferische omstandigheden over te brengen, zoals 'zonnige middag', 'bewolkt met lichte regen' of 'half bewolkt met een zacht briesje'.
Geef een relatief algemeen overzicht van het weer van morgen.
Zorg ervoor dat het rapport op een vriendelijke toon is geschreven, vergelijkbaar met die van een meteoroloog die de voorspelling op een warme en laagdrempelige manier levert. Wees creatief en gebruik variaties tijdens het schrijven van het rapport om het interessant te houden.

Wolken en neerslag:
Het rapport mag geen waarden voor bewolking of neerslag bevatten, maar mag deze alleen gebruiken om de weersomstandigheden te beschrijven. Houd rekening met de volgende parameters om de bewolking en neerslag te beoordelen en zo beschrijvende weerbeschrijvingen te verkrijgen:
- SQ (zonneschijnduur per uur): Gebruik SQ-waarden om de aanwezigheid van zonneschijn en de duur ervan te bepalen.
- N (Bewolking): Beoordeel de bewolking op basis van N-waarden (9=bovenste hemel onzichtbaar) en geef dienovereenkomstig beschrijvende weerbeschrijvingen.
- R (Regenval): Bepaal het optreden van neerslag op basis van R-waarden.
- RH (relatieve vochtigheid (neerslaghoeveelheid per uur (in 0,1 mm) (-1 voor <0,05 mm) / neerslaghoeveelheid per uur (in 0,1 mm) (-1 voor <0,05 mm)))
- O (Onweer (0=deed zich niet voor; 1=deed zich voor in het voorgaande uur))
- S (Sneeuwval (0=niet voorgekomen; 1=voorgekomen in het voorgaande uur))
- M (Mist (0=niet voorgekomen; 1=voorgekomen in het voorgaande uur))
- VV (Horizontaal zicht (Horizontaal zicht tijdens observatie (0=minder dan 100m; 1=100-200m; 2=200-300m;...; 49=4900-5000m; 50=5-6km; 56=6-7km ; 57 =7-8km ...; 79=29-30km; 81=35-40km;...; minder dan 100 m; 1=100-200 m;...; 49=4900-5000 m; 29-30 km; 80=30-35 km; 81=35-40 km;...; 89=meer dan 70 km))

Beschrijf op basis van de parameters de weersomstandigheden als volgt:
Kijk naar de waarden van zonneschijn, bewolking, regen, sneeuw, mist, zicht, onweer en mistwaarden om de weersomstandigheden als volgt te beschrijven:
- als de zonneschijnwaarden voldoende zonneschijn aangeven, omschrijf dit dan als "Zonnige middag" of "Zonnige dag".
- voor hoge bewolking en lage zonneschijnwaarden beschrijven als "grijze dag".
- als mist gelijk is aan 1 en het zicht aangeeft hoeveel mist er is, omschrijven als "Mistig".
- voor matige zonneschijn, hoge bewolking en regen gelijk aan 1, omschrijven als "Zon en buien".
- als onweer gelijk is aan 1, regen gelijk is aan 1 en zonneschijn niet 0 is met wat bewolking, omschrijf het dan als "Onweersbuien".
- voor niet-0 zonneschijn en enige bewolking, omschrijven als "Zon en cumuluswolken".
- als de regen gelijk is aan 1, verschillende neerslagwaarden en de zonneschijn laag of 0 is, wordt dit omschreven als "af en toe regen".
- als de mist gelijk is aan 1 en de zonneschijn met de tijd toeneemt, beschrijft u dit als "Mist gevolgd door zon".
- als regen gelijk is aan 1, omschrijf het dan als "Regen".
- voor veel bewolking en weinig of geen zonneschijn, omschrijven als "Grijs en nauwelijks zonnig".
- voor sommige zonneschijn is onweer gelijk aan 1, en voor sommige bewolking wordt dit beschreven als "Cumuluswolken en onweersbuien".
- als sneeuw gelijk is aan 1 en verschillende neerslagwaarden, omschrijf dit dan als "Sneeuwbuien".
- voor sommige zonneschijn is sneeuw gelijk aan 1, en verschillende neerslagwaarden worden omschreven als "zon- en sneeuwbuien".
- bij weinig of 0 zonneschijn is regen gelijk aan 1, sneeuw gelijk aan 1 en verschillende neerslagwaarden worden omschreven als "Winterbuien".


Temperatuur (temperatuur in Celsius):
Bespreek de temperatuurschommelingen gedurende de dag en analyseer de gegevens uit de kolom 'temperatuur'. Kies de schrijftemperatuurwaarden voor beide datums.

Wind:
Beschrijf de windomstandigheden met behulp van de windkrachtschaal van Beaufort, rekening houdend met de volgende kenmerken:
- DD (windrichting)
- FH (windkracht Beaufort)
Bespreek de windkrachtbeschrijvingen op basis van Beaufort-schaalwaarden (FH):
0: kalm, 1: lichte lucht, 2: lichte bries, 3: zachte bries, 4: matige bries, 5: frisse bries, 6: sterke bries, 7: bijna storm, 8: storm, 9: sterke storm, 10: Storm, 11: Hevige storm, 12: Orkaan.
Zorg ervoor dat u de windomstandigheden op een beschrijvende en levendige manier beschrijft, waarbij u de schaalbeschrijvingen van Beaufort gebruikt om de sterkte en impact van de wind over te brengen.


Tabelselectie:
Zorg ervoor dat u de juiste tabel gebruikt voor de opgegeven locatie om de weergegevens op te halen. Als de locatie Eindhoven is, gebruik dan de tabel met stationwaarde 370. Als de locatie De Bilt is, gebruik dan de tabel met stationwaarde 260. Als de locatie Amsterdam is, gebruik dan expliciet alleen de tabel 'Amsterdam' met stationswaarde 240. Als de locatie Maastricht is, gebruik dan de tabel met stationwaarde 380. En als de locatie Rotterdam is, gebruik dan de tabel met stationwaarde 344. In de tabellen staan ​​de datums in het formaat YYYYMMDD. Als u een uitgebreid weerrapport wilt maken, begint u met het ophalen van de gegevens van de huidige dag, zodat u de weersomstandigheden van vandaag kunt bespreken.
Zorg er vervolgens voor dat u ook de gegevens voor de volgende dag ophaalt, zodat u de weersvoorspelling voor morgen in het rapport kunt opnemen.

SELECT Time_Category, temperature, winddirection, beaufort, precipitation, sunshine, cloudcover, fog, rain, snow, thunderstorm FROM [location] 
WHERE date = [date] AND (hour BETWEEN 5 AND 17 OR hour BETWEEN 18 AND 23 OR hour BETWEEN 0 AND 4)
UNION
SELECT Time_Category, temperature, winddirection, beaufort, precipitation, sunshine, cloudcover, fog, rain, snow, thunderstorm 
FROM [location] 
WHERE date = [date+1]
ORDER BY Time_Category ASC
    """
    ,
    synthesize_response: bool = True,
    display_report:bool = True,
):
    
    tables = tables 
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=tables,
        synthesize_response=synthesize_response,
    )
    query_str = question + "\n" + format_hint
    try:
        response = query_engine.query(query_str)
        response_md = str(response)
        sql = response.metadata["sql_query"]
       
    except Exception as ex:
        response_md = "`⚠️ No data ⚠️`"
        sql = f"ERROR: {str(ex)}"

    if display_report:
        display(Markdown(response_template.format(
            question=question,
            response=response_md,
            sql=sql
        )))

    return question, response_md, sql, response

# call the function here
question, response_md, sql, response = chat_to_sql_dutch("location - De Bilt, date - December 10, 2023")

# COMMAND ----------

response.metadata['result']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi step approach

# COMMAND ----------

# MAGIC %md
# MAGIC ##English

# COMMAND ----------

from IPython.display import Markdown, display

response_template = """
## Question
> {question}

## Answer
{response}

## Generated SQL Query
```
{sql}
```

## SQL data
```
{sql_response}
```
"""


def chat_to_sql(
    question: str | list[str],
    tables: list[str] | None = None,
    # format_hint: str = "Please generate a weather report that gives 3 lines about overall weather of the day. And one short paragraph for morning, afternoon, and evening each",
    format_hint: str = """  
Generate a weather report in for the given location and given time in the Netherlands. Provide a comprehensive overview of the weather conditions for the specified time. Ensure that the report is written in a professinal tone, resembling that of a meteorologist delivering the forecast with warmth and approachability. Keep the information no longer than 3 sentences. Include variations while writing the report to keep it interesting. Do not mention numerical cloud cover values directly in the report.

For the location and given time, analyze the weather data to create a detailed report. Begin by summarizing the overall weather forecast for the given time in a paragraph that paints a vivid picture of what to expect. 

Follow the provided example format for the weather report:
Start by introducing the given time, and then give some info about temperature, precipitation and wind. Paint a vivid picture of what to expect. 
Ensure that the report is written in a friendly tone, resembling that of a meteorologist delivering the forecast with warmth and approachability. Be creative and include variations while writing the report to keep it interesting. 

Clouds and Precipitation:
Give an indication of how cloudy the day will be, based on cloud cover values. Instead of numerical values for cloud cover, use descriptive terms derived from the data to illustrate the cloudiness and sunshine. 
Consider the following parameters to assess precipitation and provide descriptive weather descriptions in the report accordingly:
- SQ (Duration of sunshine per hourly period): Use SQ values to determine the presence of sunshine and its duration.
- N (Cloud cover): Assess cloud cover based on N values(0=no clouds, 9=sky invisible) and provide descriptive weather descriptions accordingly.
- R (Rainfall): Determine the occurrence of rainfall based on R values.
- RH (Hourly precipitation amount)(Relative Humidity (Hourly precipitation amount (in 0.1 mm) (-1 for <0.05 mm) / Hourly precipitation amount (in 0.1 mm) (-1 for <0.05 mm)))
- O (Thunderstorm (0=did not occur; 1=occurred in the previous hour))
- S (Snowfall (0=did not occur; 1=occurred in the previous hour))
- M (Mist (0=did not occur; 1=occurred in the previous hour))

Based on the parameters, describe the sunshine and cloudiness as follows:
Look at values of sunshine, cloudcover, rain, snow, fog, thunderstorm, Relative Humidity values to describe the weather conditions as follows:
Sunny, Gray, Foggy, sun and showers, Thunderstorms, sun and cumulus clouds, occasional rain, fog followed by sun, rain, hardly sunny, cumulus clouds and thunderstorms, snow showers, sun ans snow showers, and winter showers.
- if sunshine values indicate ample sunshine, describe as "Sunny afternoon" or "Sunny day".
- for high cloudcover and low sunshine values describe as "Gray day".
- if fog equals 1 describe as "Foggy".
- for moderate sunshine, high cloudcover, and rain equals 1, describe as "Sun and showers".
- if thunderstorm equals 1, rain equals 1, and sunshine is non 0 with some cloudcover, describe as "Thunderstorms".
- for non 0 sunshine and some cloudcover, describe as "Sun and cumulus clouds".
- if rain equals 1, various precipitation values, and sunshine is low or 0, describe as "Occasional rain".
- if fog equals 1 and sunshine increases with time describe as "Fog followed by sun".
- for rain equals 1, describe as "Rain".
- for high cloudcover and low or no sunshine describe as "Gray and hardly sunny".
- for some sunshine, thunderstorm equals 1, and some cloudcover, describe as "Cumulus clouds and thunderstorms".
- if snow equals 1, and various precipitation values, describe as "Snow showers".
- for some sunshine, snow equals 1, and various precipitation values describe as "Sun and snow showers".
- if low or 0 sunshine, rain equals 1, snow equals 1, and various precipitation values describe as "Winter showers".



Temperature (temperature in celsius):
Discuss the temperature variations throughout the day, analyzing data from the "temperature" column. Pick the right temperature values for both the dates. 

Wind:
Describe the wind conditions using the Beaufort wind force scale, taking into account the following features:
- DD (Wind Direction)
- FH (Beaufort wind force)
Discuss the wind force descriptions based on Beaufort scale values (FH):
0: Calm, 1: Light Air, 2: Light Breeze, 3: Gentle Breeze, 4: Moderate Breeze, 5: Fresh Breeze, 6: Strong Breeze, 7: Near Gale, 8: Gale, 9: Strong Gale, 10: Storm, 11: Violent Storm, 12: Hurricane.
Ensure to describe wind conditions in a descriptive and vivid manner, utilizing the Beaufort scale descriptions to convey the strength and impact of the wind.

Table Selection:
Ensure to utilize the correct table named after the location requested by the user to fetch the weather data. If the location is Eindhoven, use the table with station value 370. If the location is De Bilt, use the table with station value 260. If the location is Amsterdam, use the table with station value 240. If the location is Maastricht, use the table with station value 380. And if the location is Rotterdam, use the table with station value 344. The dates are in the format YYYYMMDD in the tables. 
Please consider the following hour for each section of the report:
When talking about today, look at the data only for the given date and given location.
- When talking about today morning, look at the rows with Time_category value "Morning" for the given date and given location.
- When talking about today afternoon, look at the rows with Time_category value "Afternoon" for the given date and given location.
- When talking about today Evening, look at the rows with Time_category value "Evening" for the given date and given location.
- When talking about today Night, look at the rows with Time_category value "Night" for the given date and given location.
- When talking about the full day, look at all Time_category values that are not Null for the given day and given location.

SELECT Time_Category, temperature, winddirection, beaufort, precipitation, sunshine, cloudcover, fog, rain, snow, thunderstorm FROM [location] 
WHERE date = [date] AND Time_Category = [time_category]

or

SELECT Time_Category, temperature, winddirection, beaufort, precipitation, sunshine, cloudcover, fog, rain, snow, thunderstorm FROM [location] 
WHERE date = [date] OR Time_Category IS NOT NULL

    """
    ,
    synthesize_response: bool = True,
    print = False
):
    
    tables = tables 
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=tables,
        synthesize_response=synthesize_response,
    )
    query_str = question + "\n" + format_hint
    try:
        response = query_engine.query(query_str)
        response_md = str(response)
        sql = response.metadata["sql_query"]
    except Exception as ex:
        response_md = "`⚠️ No data ⚠️`"
        sql = f"ERROR: {str(ex)}"

    if print:
        display(Markdown(response_template.format(
            question=question,
            response=response_md,
            sql=sql,
            sql_response = response.metadata['result']
        )))
    # return response_md
    return {"response_md": response_md, "sql": sql, "sql_response": response.metadata['result']}

location_date = "location - Maastricht, date - July 19, 2023"
location_date_tomorrow = "location - Maastricht, date - July 20, 2023"
  

# call the function here
morning = chat_to_sql(f"{location_date}, time - Morning",print=True)
afternoon = chat_to_sql(f"{location_date}, time - Afternoon",print=True)
evening = chat_to_sql(f"{location_date}, time - Evening",print=True)
night = chat_to_sql(f"{location_date}, time - Night",print=True)
tomorrow = chat_to_sql(f"{location_date_tomorrow}, time - full day",print=True)



# COMMAND ----------

# MAGIC %md
# MAGIC ##Dutch

# COMMAND ----------

# from IPython.display import Markdown, display

# response_template = """
# ## Question
# > {question}

# ## Answer
# {response}

# ## Generated SQL Query
# ```
# {sql}
# ```

# ## SQL data
# ```
# {sql_response}
# ```
# """


# def chat_to_sql_dutch(
#     question: str | list[str],
#     tables: list[str] | None = None,
#     # format_hint: str = "Please generate a weather report that gives 3 lines about overall weather of the day. And one short paragraph for morning, afternoon, and evening each",
#     format_hint: str = """  

# Genereer een weerbericht voor de opgegeven locatie en bepaalde tijd in Nederland. Geef een uitgebreid overzicht van de weersomstandigheden voor de opgegeven tijd. Zorg ervoor dat het rapport op een professionele toon is geschreven, vergelijkbaar met die van een meteoroloog die de voorspelling op een warme en laagdrempelige manier levert. Maak de informatie niet langer dan 3 zinnen. Varieer ook tijdens het schrijven van het weerbericht, dan blijft het gevarieerd. Gebruik geen numerieke N-waarde in het weerbericht.

# Analyseer voor de locatie en het opgegeven tijdstip de weergegevens om een ​​gedetailleerd weerbericht te maken. Begin met het samenvatten van de algemene weersvoorspelling voor een bepaalde tijd in een paragraaf die een levendig beeld schetst van wat u kunt verwachten.

# Volg de onderstaande structuur voor het weerbericht:
# Begin met het introduceren van de gevraagde moment van de dag, of in het geval van een hele dag, de datum en geef vervolgens wat informatie over temperatuur, neerslag en wind. Voeg meer informatie toe om een ​​levendig beeld te schetsen van wat u kunt verwachten.

# Wolken en neerslag:
# Geef op basis van de bewolking een woordelijke indicatie van hoe bewolkt de dag zal zijn, zoals 'zeer bewolkte dag' of 'strakblauwe lucht'. Kijk naar de onderstaande informatie om de juiste termen voor de gegeven waarden te vinden. Houd rekening met de volgende parameters om de bewolking en neerslag te beoordelen en geef bijpassende beschrijvende weerbeschrijvingen in het rapport:
# - SQ (Zonneschijnduur per uurperiode): Gebruik SQ-waarden om de aanwezigheid van zonneschijn en de duur ervan te bepalen.
# - N (Bewolking): Beoordeel de bewolking op basis van N-waarden (0=geen wolken, 9=lucht onzichtbaar) en geef bijpassende beschrijvende weerbeschrijvingen, zoals "een grijze dag" of "her en der wat bewolking" of "flink bewolkt".
# - R (Regenval): Bepaal het optreden van neerslag op basis van R-waarden.
# - RH (relatieve vochtigheid (neerslaghoeveelheid per uur (in 0,1 mm) (-1 voor <0,05 mm) / neerslaghoeveelheid per uur (in 0,1 mm) (-1 voor <0,05 mm)))
# - O (Onweer (0=deed zich niet voor; 1=deed zich voor in het voorgaande uur)) Noem dit alleen als het minstens 1 keer is voorgekomen
# - S (Sneeuwval (0=niet voorgekomen; 1=voorgekomen in het voorgaande uur)) Noem dit alleen als het minstens 1 keer is voorgekomen
# - M (Mist (0=niet voorgekomen; 1=voorgekomen in het voorgaande uur)) Noem dit alleen als het minstens 1 keer is voorgekomen

# Beschrijf de zonneschijn op basis van de parameters als volgt:
# Kijk naar de waarden van zonneschijn, bewolking, regen, sneeuw, mist, onweer en relatieve vochtigheidswaarden om de weersomstandigheden als volgt te beschrijven:
# Zonnig, Grijs, Mistig, zon en buien, Onweersbuien, zon en stapelwolken, af en toe regen, mist gevolgd door zon, regen, nauwelijks zonnig, stapelwolken en onweersbuien, sneeuwbuien, zon en sneeuwbuien, en winterbuien.
# - als de zonneschijnwaarden voldoende zonneschijn aangeven, omschrijf dit dan als "Zonnige middag" of "Zonnige dag".
# - voor hoge bewolking en lage zonneschijnwaarden omschrijven als "Grijze dag".
# - als 'fog' gelijk is aan 1, omschrijf het dan als "Mistig".
# - voor matige zonneschijn, hoge bewolking en regen gelijk aan 1, omschrijven als "Zon en buien".
# - als 'thunderstorm' gelijk is aan 1, regen gelijk is aan 1 en de zonneschijn niet 0 is met enige bewolking, omschrijf dit dan als "Onweer".
# - voor niet-0 zonneschijn en enige bewolking, omschrijven als "Zon en cumuluswolken".
# - als regen gelijk is aan 1, verschillende neerslagwaarden en de zonneschijn laag of 0 is, omschrijven als "Af en toe regen".
# - als fog gelijk is aan 1 en de zonneschijn met de tijd toeneemt, beschrijft u dit als "Mist gevolgd door zon".
# - als rain gelijk is aan 1, omschrijf het dan als "Regen".
# - voor veel bewolking en weinig of geen zonneschijn, omschrijven als "Grijs en nauwelijks zonnig".
# - voor een beetje zonneschijn en onweer gelijk aan 1, en voor wat bewolking wordt dit omschreven als "Cumuluswolken en onweersbuien".
# - als 'snow' gelijk is aan 1, en verschillende neerslagwaarden, omschrijven als "Sneeuwbuien".
# - voor een beetje zonneschijn en 'snow' gelijk aan 1, en verschillende neerslagwaarden worden omschreven als "Zon- en sneeuwbuien".
# - bij zonneschijn 0 of erg laag en regen gelijk aan 1, sneeuw gelijk aan 1 en worden verschillende neerslagwaarden beschreven als "Winterbuien".

# Let op: sneeuw, mist en onweer zijn verboden woorden tenzij ze echt voorspeld zijn.


# Temperatuur (temperatuur in Celsius):
# Bespreek de temperatuurschommelingen gedurende de dag en analyseer de gegevens uit de kolom 'temperature'. Kies de juiste temperatuurwaarden voor beide datums.

# Wind:
# Beschrijf de windomstandigheden met behulp van de windkrachtschaal van Beaufort, rekening houdend met de volgende kenmerken:
# -DD ​​(Windrichting)
# - FH (Beaufort-windkracht)
# Bespreek de windkrachtbeschrijvingen op basis van schaalwaarden van Beaufort (FH):
# 0: kalm, 1: lichte lucht, 2: lichte bries, 3: zachte bries, 4: matige bries, 5: frisse bries, 6: sterke bries, 7: bijna storm, 8: storm, 9: sterke storm, 10: Storm, 11: Hevige storm, 12: Orkaan.
# Zorg ervoor dat u de windomstandigheden op een beschrijvende en levendige manier beschrijft, waarbij u de schaalbeschrijvingen van Beaufort gebruikt om de sterkte en impact van de wind over te brengen.


# Let op: het weerbericht moet in het Nederlands geschreven worden.

# Tabelselectie:
# Zorg ervoor dat u de juiste tabel gebruikt, genoemd naar de door de gebruiker gevraagde locatie, om de weergegevens op te halen. Als de locatie Eindhoven is, gebruik dan de tabel met stationwaarde 370. Als de locatie De Bilt is, gebruik dan de tabel met stationwaarde 260. Als de locatie Amsterdam is, gebruik dan de tabel met stationwaarde 240. Als de locatie Maastricht is, gebruik dan de tabel met stationwaarde 380. En als de locatie Rotterdam is, gebruik dan de tabel met stationwaarde 344. In de tabellen staan ​​de datums in het formaat YYYYMMDD.
# Houd voor elk onderdeel van het rapport rekening met het volgende uur:
# Als je het over vandaag hebt, kijk dan alleen naar de gegevens voor de gegeven datum en bepaalde locatie.
# - Als je het over vandaag 's ochtends hebt, kijk dan naar de rijen met Time_category-waarde 'Morning' voor de opgegeven datum en opgegeven locatie.
# - Als je het over vandaag 's middags hebt, kijk dan naar de rijen met Time_category-waarde 'Afternoon' voor de opgegeven datum en opgegeven locatie.
# - Als je het over vandaag 's avonds hebt, kijk dan naar de rijen met Time_category-waarde 'Evening' voor de opgegeven datum en opgegeven locatie.
# - Als je het over vandaag Nacht hebt, kijk dan naar de rijen met Time_category waarde 'Night' voor de gegeven datum en gegeven locatie.
# - Als het over de 'full day' gaat, kijk dan naar alle Time_category-waarden die niet Null zijn voor de gegeven dag en gegeven locatie.

# SELECT Time_Category, temperature, winddirection, beaufort, precipitation, sunshine, cloudcover, fog, rain, snow, thunderstorm FROM [location] 
# WHERE date = [date] AND Time_Category = [time_category]

# or

# SELECT Time_Category, temperature, winddirection, beaufort, precipitation, sunshine, cloudcover, fog, rain, snow, thunderstorm FROM [location] 
# WHERE date = [date] OR Time_Category IS NOT NULL

#     """
#     ,
#     synthesize_response: bool = True,
#     print: bool = False
# ):
    
#     tables = tables 
#     query_engine = NLSQLTableQueryEngine(
#         sql_database=sql_database,
#         tables=tables,
#         synthesize_response=synthesize_response,
#     )
#     query_str = question + "\n" + format_hint
#     try:
#         response = query_engine.query(query_str)
#         response_md = str(response)
#         sql = response.metadata["sql_query"]
        
#     except Exception as ex:
#         response_md = "`⚠️ No data ⚠️`"
#         sql = f"ERROR: {str(ex)}"  
#     if print:
#         display(Markdown(response_template.format(
#             question=question,
#             response=response_md,
#             sql=sql,
#             sql_response = response.metadata['result']
#         )))
#     return response_md

# location_date = "location - De Bilt, date - January 19, 2023"
# location_date_tomorrow = "location - De Bilt, date - January 20, 2023"
  

# # call the function here
# morning = chat_to_sql_dutch(f"{location_date}, time - Morning",print=True)
# afternoon = chat_to_sql_dutch(f"{location_date}, time - Afternoon",print=True)
# evening = chat_to_sql_dutch(f"{location_date}, time - Evening",print=True)
# night = chat_to_sql_dutch(f"{location_date}, time - Night",print=True)
# tomorrow = chat_to_sql_dutch(f"{location_date_tomorrow}, time - full day",print=True)

# COMMAND ----------

context = morning + afternoon + evening + night + tomorrow
print(context)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combining the individual report sections

# COMMAND ----------

# MAGIC %md
# MAGIC ##English

# COMMAND ----------

# import openai 
# from openai import OpenAI
# from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# os.environ['OPENAI_API_KEY'] = dbutils.secrets.get('diw-kv-di-we-di', 'test-ds-chatgpt-key')

# openai.api_type = "azure"
# openai.api_base = "https://test-ds-chatgpt.openai.azure.com/"
# openai.api_version = "2023-03-15-preview"
# openai.api_key = os.getenv("OPENAI_API_KEY")

# token_provider = get_bearer_token_provider(openai.api_key, "https://cognitiveservices.azure.com/.default")


# from openai import AzureOpenAI

# client = AzureOpenAI(
#   api_key = openai.api_key,  
#   api_version = "2024-02-01",
#   azure_endpoint = openai_api_base
# )


# prompt_stitch = f"""

# For the location and day, analyze the weather data to create a detailed report. Do not mention dates in the report. 

# Note: Focus on highlighting any changes in weather conditions throughout the day. If conditions remain consistent, avoid repeating details unnecessarily. Skip the description of cloud cover and do not mention precipitation in every section if there is no precipitation at all during the day.

# The morning is described here: 
# {morning}

# The afternoon is described here: 
# {afternoon}

# The evening is described here: 
# {evening}

# The night is described here:
# {night}

# Tomorrow is described here:
# {tomorrow}

# Begin by summarizing the overall weather forecast in 2 sentences that paints a vivid picture of what to expect without mentioning specific time of the day. This should be followed by a paragraph for the morning, starting with "This morning", where you can bring to life the atmospheric conditions. Subsequently, provide a paragraph for the afternoon, starting with "This afternoon", continuing the friendly tone. Then, describe the evening and night together in a paragraph capturing the essence of the nighttime weather in an engaging manner. Finally, offer a few lines about the next day, starting with "Tomorrow", keeping the tone light and the descriptions lively. 

# """

# def get_response(messages, max_tokens=800, temperature = 0.6):
#     response = client.chat.completions.create(
#                                         model="gpt-35-turbo",
#                                         #engine="gpt-35-turbo",
#                                         messages=messages,
#                                         temperature=temperature, # 1
#                                         max_tokens = max_tokens
#                                         )
#     return  response.choices[0].message.content

# def append_comment(messages, comment):
#   messages.append({"role":"user","content":comment})
#   return messages
# messages = append_comment([{"role": "system", "content": "You're meteorologist writing for buienradar"}],prompt_stitch)
# # print(messages)

# print(get_response(messages))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Dutch

# COMMAND ----------

import openai 
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

os.environ['OPENAI_API_KEY'] = dbutils.secrets.get('diw-kv-di-we-di', 'test-ds-chatgpt-key')

openai.api_type = "azure"
openai.api_base = "https://test-ds-chatgpt.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

token_provider = get_bearer_token_provider(openai.api_key, "https://cognitiveservices.azure.com/.default")


from openai import AzureOpenAI

client = AzureOpenAI(
  api_key = openai.api_key,  
  api_version = "2024-02-01",
  azure_endpoint = openai_api_base
)

prompt_sum = f"""
Vertaal voor deze locatie en dag de weergegevens om een ​​gedetailleerd weerbericht te maken. Vermeld geen datums in het weerbericht. 

Opmerking: Focus op het benadrukken van eventuele veranderingen in de weersomstandigheden gedurende de dag. Als de omstandigheden consistent blijven, vermijd dan het onnodig herhalen van details. Geef geen numerieke waarden voor de bewolking. 

Hieronder vindt je de Engelse weerberichten:
De ochtend wordt hier beschreven:
{morning}

De middag wordt hier beschreven:
{afternoon}

De avond wordt hier beschreven:
{evening}

De nacht wordt hier beschreven:
{night}

De volgende dag wordt hier beschreven:
{tomorrow}

Begin met het samenvatten van de algemene weersvoorspelling in twee zinnen, zodat de lezer een levendig beeld krijgt van wat hij kan verwachten, zonder een specifiek tijdstip van de dag te noemen. Dit moet worden gevolgd door een paragraaf voor de ochtend, beginnend met "In de ochtend", waarin de weersomstandigheden tot leven worden gebracht. Geef vervolgens een paragraaf voor de middag, beginnend met "Vanmiddag" of "In de middag", en beschrijf dit op dezelfde luchtige beschrijvende toon. Beschrijf vervolgens de avond en de nacht samen in een paragraaf waarin je op een boeiende manier de het weer in de avond en nacht beschrijft. Vertel ten slotte een paar regels over de volgende dag, te beginnen met 'Morgen', waarbij u de toon licht en de beschrijvingen levendig houdt. 
"""




# COMMAND ----------

def get_response(messages, max_tokens=800, temperature = 0.0):
    response = client.chat.completions.create(
                                        model="gpt-35-turbo",
                                        #engine="gpt-35-turbo",
                                        messages=messages,
                                        temperature=temperature, # 1
                                        max_tokens = max_tokens
                                        )
    return  response.choices[0].message.content

def append_comment(messages, comment):
  messages.append({"role":"user","content":comment})
  return messages

messages = append_comment([{"role": "system", "content": "Je bent een meteoroloog in dienst van buienradar en vertaalt Engelse weerberichten naar het Nederlands in de stijl van buienradar"}],prompt_sum)
# print(messages)

print(get_response(messages))

# COMMAND ----------

# MAGIC %md
# MAGIC #Generating a number of reports sequentially for Evaluation purposes

# COMMAND ----------

import random
import pandas as pd
from datetime import datetime, timedelta

# def generate_random_date():
#     start_date = datetime(2023, 1, 1)
#     end_date = datetime(2023, 12, 31)

#     random_date = start_date + (end_date - start_date) * random.random()

#     return random_date

def generate_random_date():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    return random_date

def format_date(date):
    return date.strftime("%B %d, %Y")


# Generate the first random date
date = generate_random_date()
print(format_date(date))

# Generate the next date by adding one day to the first random date
next_date = date + timedelta(days=1)
print(format_date(next_date))


def get_response(messages, max_tokens=800, temperature = 0.0):
    response = client.chat.completions.create(
                                        model="gpt-35-turbo",
                                        #engine="gpt-35-turbo",
                                        messages=messages,
                                        temperature=temperature, # 1
                                        max_tokens = max_tokens
                                        )
    return  response.choices[0].message.content

def append_comment(messages, comment):
  messages.append({"role":"user","content":comment})
  return messages

# COMMAND ----------

locations = ["Rotterdam", "De Bilt", "Eindhoven", "Maastricht", "Amsterdam"]
reports_dictionaries = [] 

# Open a text file to store the results
with open("weather_reports.txt", "w") as file:
    for _ in range(400):
        # Select a random location and date
        location = random.choice(locations)
        rand_date = generate_random_date()
        date = format_date(rand_date)
        today = f"location - {location}, date - {date}"
        next_rand_date = rand_date + timedelta(days=1)
        next_date = format_date(next_rand_date) 
        next_day = f"location - {location}, date - {next_date}"
      
        # Call the chat_to_sql_dutch function with the random location and date
        morning = chat_to_sql(f"{today}, time - Morning",print=False)
        sql_response_morning = morning['sql_response']
        morning_report = morning['response_md']
        afternoon = chat_to_sql(f"{today}, time - Afternoon",print=False)
        sql_response_noon = afternoon['sql_response']
        noon_report = afternoon['response_md']
        evening= chat_to_sql(f"{today}, time - Evening",print=False)
        sql_response_evening = evening['sql_response']
        evening_report = evening['response_md']
        night = chat_to_sql(f"{today}, time - Night",print=False)
        sql_response_night = night['sql_response']
        night_report = night['response_md']
        tomorrow = chat_to_sql(f"{next_day}, time - full day",print=False)
        sql_response_tomorrow = tomorrow['sql_response']
        tomorrow_report = tomorrow['response_md']

        prompt_sum = f"""
        Vertaal voor deze locatie en dag de weergegevens om een ​​gedetailleerd weerbericht te maken. Vermeld geen datums in het weerbericht. 

        Opmerking: Focus op het benadrukken van eventuele veranderingen in de weersomstandigheden gedurende de dag. Als de omstandigheden consistent blijven, vermijd dan het onnodig herhalen van details. Geef geen numerieke waarden voor de bewolking. 

        Hieronder vindt je de Engelse weerberichten:
        De ochtend wordt hier beschreven:
        {morning_report}

        De middag wordt hier beschreven:
        {noon_report}

        De avond wordt hier beschreven:
        {evening_report}

        De nacht wordt hier beschreven:
        {night_report}

        De volgende dag wordt hier beschreven:
        {tomorrow_report}

        Begin met het samenvatten van de algemene weersvoorspelling in twee zinnen, zodat de lezer een levendig beeld krijgt van wat hij kan verwachten, zonder een specifiek tijdstip van de dag te noemen. Dit moet worden gevolgd door een paragraaf voor de ochtend, beginnend met "In de ochtend", waarin de weersomstandigheden tot leven worden gebracht. Geef vervolgens een paragraaf voor de middag, beginnend met "Vanmiddag" of "In de middag", en beschrijf dit op dezelfde luchtige beschrijvende toon. Beschrijf vervolgens de avond en de nacht samen in een paragraaf waarin je op een boeiende manier de het weer in de avond en nacht beschrijft. Vertel ten slotte een paar regels over de volgende dag, te beginnen met 'Morgen', waarbij u de toon licht en de beschrijvingen levendig houdt.
        """

        # --Debugging (bitch takes the same location in every iteration)
        # print(f"Prompt for {location} on {today}:\n{prompt_sum}\n")

        messages = append_comment(
            [{"role": "system", "content": "Je bent een meteoroloog in dienst van buienradar en vertaalt Engelse weerberichten naar het Nederlands in de stijl van buienradar"}],
            prompt_sum
        )

        response = get_response(messages)

        # Save the responses
        report_entry = {
            "location": location,
            "date": today,
            "report": response,
            "sql_response_morning": sql_response_morning,
            "sql_response_afternoon": sql_response_noon,
            "sql_response_evening": sql_response_evening,
            "sql_response_night": sql_response_night,
            "sql_response_tomorrow": sql_response_tomorrow
        }
        reports_dictionaries.append(report_entry)
        file.write(f"{report_entry}\n")

df = pd.DataFrame(reports_dictionaries)
display(df)

# COMMAND ----------

import urllib.parse


def sql_to_html_table(sql_responses, period):
    html = f"<h3>{period}</h3>"
    html += "<table border='1'>"
    html += "<tr><th>Time</th><th>Temperature</th><th>winddirection</th><th>beaufort</th><th>precipitation</th><th>sunshine</th><th>cloudcover</th><th>fog</th><th>rain</th><th>snow</th><th>thunderstorm</th></tr>"
    for response in sql_responses:
        html += "<tr>" + "".join([f"<td>{item}</td>" for item in response]) + "</tr>"
    html += "</table>"
    return html

# Function to combine all the columns into a single HTML column
def combine_columns_to_html(row):
    html = f"<html> <p><strong>Location:</strong> {row['location']}</p>"
    html += f"<p><strong>Date:</strong> {row['date']}</p>"
    html += f"<p><strong>Report:</strong> {row['report']}</p>"
    
    html += sql_to_html_table(row['sql_response_morning'], "Morning")
    html += sql_to_html_table(row['sql_response_afternoon'], "Afternoon")
    html += sql_to_html_table(row['sql_response_evening'], "Evening")
    html += sql_to_html_table(row['sql_response_night'], "Night")
    html += sql_to_html_table(row['sql_response_tomorrow'], "Tomorrow")
     
    html += "</html>"
    data_uri = create_data_uri(html)
    html_link = f'<a href="{data_uri}" target="_blank">View Report</a>'

    return html

  
def create_data_uri(html_content):
    html_escaped = urllib.parse.quote(html_content)
    return f"data:text/html;charset=utf-8,{html_escaped}"
  
# df['combined_html'] = df.apply(combine_columns_to_html, axis=1)

# data_uri = create_data_uri(df['combined_html'][0])

# html_link = f'<a href="{data_uri}" target="_blank">View Report</a>'
# print(html_link)

df['html_report'] = df.apply(combine_columns_to_html, axis=1)
df['html_report']

# COMMAND ----------

df['external_id'] = df.index.astype(str) + "_23052024"
def write_html_file(row):
  filename = f"/dbfs/mnt/labelbox-datasets/buienradarv2/{row['external_id']}.HTML"
  with open(filename, "w", encoding="utf-8") as file:
    file.write(row['html_report'])
  return filename

df['filename'] = df.apply(lambda row: write_html_file(row),axis=1)
df.to_csv('/dbfs/FileStore/megha/Reports_Dataset.csv', index=False)

# COMMAND ----------



# COMMAND ----------

# pd.set_option('display.max_colwidth', None) # to see the full text
pd.reset_option('display.max_colwidth') # to see truncated result

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #Extraaaaa random stuff!!

# COMMAND ----------

import random
from datetime import datetime

def generate_random_date():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    random_date = start_date + (end_date - start_date) * random.random()

    # Format the random date as "Month Day, Year"
    formatted_date = random_date.strftime("%B %d, %Y")

    return formatted_date


# COMMAND ----------

import random
import pandas as pd
from datetime import datetime, timedelta

locations = ["Rotterdam", "De Bilt", "Eindhoven", "Maastricht", "Amsterdam"]
# dates = ["December 2, 2023", "December 4, 2023", "December 8, 2023", "December 11, 2023", "December 14, 2023", "December 17, 2023", "December 22, 2023", "December 24, 2023", "December 29, 2023"]


reports_dictionaries = [] 

# Open a text file to store the results
with open("weather_reports.txt", "w") as file:
    for _ in range(400):
        # Select a random location and date
        location = random.choice(locations)
        date = generate_random_date()
        
        # Call the chat_to_sql_dutch function with the random location and date
        question, response, sql_query = chat_to_sql_dutch(f"location - {location}, date - {date}",display_report=False)
        reports_dictionaries += [{"question":question,"response":response,"sql_query":sql_query}]


df = pd.DataFrame(reports_dictionaries)
display(df)

# COMMAND ----------

df.to_csv('/dbfs/FileStore/megha/Reports_Dataset.csv', index=False)

# COMMAND ----------

query_str = "Fetch the data for first day of January, 2023 in De Bilt?"
response = sql_query_engine.query(query_str)
# response
context_weather_data = response.source_nodes[0].text


# COMMAND ----------

# MAGIC %md
# MAGIC ###Fetching retrieved index in Pinecone

# COMMAND ----------


import numpy as np

question_embedding = embed_model.aget_query_embedding(query_str)
question_embedding = np.array(question_embedding)
# print(question_embedding.tolist())
# query_emb = question_embedding[:5]
# print(query_emb)


# COMMAND ----------

import numpy as np
from nomic import atlas
import nomic

nomic.login('nk-y3AUruzjsu9ItHuKvNHQUyvhfRBoHlgMqPVADM_uW74')

# COMMAND ----------

#Demo
# num_embeddings = 100
# embedding_dim = 50
# dummy_embeddings = np.random.rand(num_embeddings, embedding_dim)

# dummy_data = {
#     'id': range(num_embeddings),  # Assuming 'id' is a unique identifier for each data point
#     'Score': np.random.randint(1, 10, size=num_embeddings)  # Random 'Score' values for each data point
# }
# df = pd.DataFrame(dummy_data)

# # Create an Atlas project and map the embeddings
# project = atlas.map_data(embeddings=dummy_embeddings,
#                                data=df.to_dict('records'),
#                                id_field='id',
#                                )
# map = project.maps[0]

# map


# COMMAND ----------

# MAGIC %md
# MAGIC store the table schema in an index so that during query time we can retrieve the right schema.

# COMMAND ----------

from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core import VectorStoreIndex


table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    (SQLTableSchema(table_name="De_Bilt")),
    (SQLTableSchema(table_name="eindhoven"))
]  # add a SQLTableSchema for each table

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=1)
)

# COMMAND ----------

response = query_engine.query("how many days it rained in February, 2023 in Eindhoven?")
response

# COMMAND ----------

response = query_engine.query("What was the average temp in De Bilt in January 2023? convert the temp in celsius")
response

# COMMAND ----------

response = query_engine.query("which was the coldest day in De Bilt? Also give me the month and temp of that day")
response

# COMMAND ----------

desc = """
Generate a weather report for the location asked in the user query. Write the report in a thug style and it can be upto 3 sentences long.
"""

city_stats_text = (
    "The table gives information about different weather parameters of the city"
    " given city.\nThe user will query for weather report of particular location"
)

table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    (SQLTableSchema(table_name="De_Bilt", context_str=city_stats_text + desc))
]

# COMMAND ----------

from llama_index.core.retrievers import NLSQLRetriever


nl_sql_retriever = NLSQLRetriever(
    sql_database, tables=table_names, return_raw=True
)

# COMMAND ----------

results = nl_sql_retriever.retrieve(
    "Can you generate the weather report for first day of January, 2023 in De Bilt??"
)
results

# COMMAND ----------

# MAGIC %md
# MAGIC Set up Query Engines

# COMMAND ----------

from llama_index.core.query_engine import SQLAutoVectorQueryEngine, RetrieverQueryEngine
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.indices.vector_store import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

# COMMAND ----------

sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    description=(
        "Useful for translating a natural language query into an SQL query over tables containing:"
        "weather_data, containing weather information for Netherlands."
    ),
)



