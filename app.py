# app.py
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Load and clean the dataset
def load_and_clean_data():
    df = pd.read_csv("hotel_bookings.csv")
    
    # Handle missing values
    df.fillna({"children": 0, "country": "Unknown", "agent": 0, "company": 0}, inplace=True)
    
    # Convert date columns to datetime
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])
    
    # Remove outliers or invalid rows
    df = df[df["adr"] > 0]  # Remove rows with negative average daily rate
    return df

df = load_and_clean_data()

# Analytics functions
def revenue_trends(df):
    df["revenue"] = df["adr"] * (df["stays_in_weekend_nights"] + df["stays_in_week_nights"])
    monthly_revenue = df.resample('M', on='reservation_status_date')["revenue"].sum()
    monthly_revenue.plot(title="Monthly Revenue Trends")
    plt.show()

def cancellation_rate(df):
    total_bookings = df.shape[0]
    cancelled_bookings = df[df["is_canceled"] == 1].shape[0]
    cancellation_rate = (cancelled_bookings / total_bookings) * 100
    print(f"Cancellation Rate: {cancellation_rate:.2f}%")

def geographical_distribution(df):
    country_distribution = df["country"].value_counts().head(10)
    country_distribution.plot(kind="bar", title="Top 10 Countries by Bookings")
    plt.show()

def lead_time_distribution(df):
    df["lead_time"].plot(kind="hist", bins=30, title="Booking Lead Time Distribution")
    plt.show()

# RAG-based Q&A with GPT-Neo
CACHE_DIR = "./model_cache"  # Directory to cache the model
os.makedirs(CACHE_DIR, exist_ok=True)

# Load the tokenizer and model
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"  # Smaller model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

# Load the SentenceTransformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the dataset
embeddings = embedding_model.encode(df.astype(str).to_numpy())

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def summarize_context(relevant_data):
    """
    Summarize the relevant data into a concise context.
    """
    # Count bookings by country
    country_counts = relevant_data["country"].value_counts().to_dict()
    summary = "Bookings by country:\n"
    for country, count in country_counts.items():
        summary += f"- {country}: {count} bookings\n"
    return summary

def rag_qa(question):
    # Retrieve relevant data using FAISS
    query_embedding = embedding_model.encode([question])
    distances, indices = index.search(query_embedding, k=19000)  # Retrieve top 2000 relevant rows
    relevant_data = df.iloc[indices[0]]
    
    # Summarize the context
    context = summarize_context(relevant_data)
    
    # Prepare the prompt
    prompt = (
        f"Answer the question based on the following context:\n"
        f"Context:\n{context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    
    # Tokenize the prompt and truncate if necessary
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate the answer using GPT-Neo
    outputs = model.generate(**inputs, max_new_tokens=50)  # Limit the output to 50 tokens
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-process the answer to remove repetitions
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()
    return answer

# FastAPI app
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/analytics")
def get_analytics():
    revenue_trends(df)
    cancellation_rate(df)
    geographical_distribution(df)
    lead_time_distribution(df)
    return {"message": "Analytics generated and displayed."}

@app.post("/ask")
def ask_question(query: Query):
    question = query.question
    answer = rag_qa(question)
    return {"question": question, "answer": answer}

@app.get("/health")
def health_check():
    return {"status": "healthy", "dependencies": ["FAISS", "GPT-Neo", "FastAPI"]}

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)