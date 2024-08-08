# WeatherReportingSystem

An automated system for generating location-specific weather reports.

This project utilizes Retrieval Augmented Generation (RAG) to analyze and interpret local weather data to generate comprehensive and informative weather reports. Leveraging advanced AI techniques, it automates the creation of localized weather reports, addressing the labor-intensive process currently used by meteorologists.

## Files Description

- **Report_generation_pipeline.py**: Uses the Retrieval Augmented Generation (RAG) approach to generate local weather reports.
- **Evaluation.py**: Evaluates the results and analyzes user perceptions of AI-generated weather reports.
- **weather_assistant_chatbot.py**: Implements a chatbot to provide a personalized weather chat experience.

## Technologies Used

- **Language Model**: Azure OpenAI's GPT-3.5-turbo
- **Embeddings**: Azure OpenAI's text-embedding-ada-002 model
- **Vector Database**: Pinecone
