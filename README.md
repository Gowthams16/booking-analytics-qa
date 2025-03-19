# Hotel Booking Analytics & QA System

## **Features**

1. **Data Analytics**:

   - Revenue trends over time.
   - Cancellation rate as a percentage of total bookings.
   - Geographical distribution of users.
   - Booking lead time distribution.

2. **Retrieval-Augmented Question Answering (RAG)**:

   - Answer user questions about the booking data using GPT-Neo.

3. **REST API**:
   - Endpoints for analytics and question answering.

## **Installation and Run**

### **Prerequisites**

- Python 3.8 or higher.
- Git (to clone the repository).
- A Hugging Face account (to access GPT-Neo).

### **Steps to Set Up and Run**

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Gowthams16/booking-analytics-qa.git
   cd booking-analytics-qa

   ```

2. **Create a Virtual Environment**:
   python -m venv venv

3. **Activate the Virtual Environment**:
   Windows: venv\Scripts\activate
   Mac/Linux: source venv/bin/activate

4. **Install Dependencies**:
   pip install -r requirements.txt

5. **Login to Hugging Face**:
   huggingface-cli login

6. **Running the Application**:
   python app.py
   (The server will start at http://127.0.0.1:8000 or at http://0.0.0.0:8000)

### **API Endpoints**

POST: http://0.0.0.0:8000/analytics
Expected Result: Generates and displays analytics reports.

POST: http://0.0.0.0:8000/ask
Expected Result:
{
"question": "Which country has the highest bookings?",
"answer": "Answer the question based on the following context:\nContext:\nBookings by country:\n- PRT: 10353 bookings\n-
......
}

GET: http://0.0.0.0:8000/health
Expected Result:
{
"status": "healthy",
"dependencies": [
"FAISS",
"GPT-Neo",
"FastAPI"
]
}

**Check the results folder to see the screenshots of the results**
