# 🌤️ Weather Agent API with Redis & FastAPI

## 🔍 Overview

**Weather Agent API** is a **powerful AI-driven FastAPI service** designed to deliver intelligent, contextual, and real-time weather insights. Backed by **LangChain**, **Google Gemini**, and **Redis**, it offers:

* 🌦️ Instant weather updates for any city
* 🌍 Side-by-side city weather comparisons
* 💬 Conversational memory through persistent sessions
* 🧠 General knowledge Q\&A powered by generative AI

This solution is ideal for both developers and end-users looking for a smart, responsive weather chatbot interface with modern AI capabilities.

---

## 🚀 Key Features

* ⚡ **FastAPI Backend** — High-performance async REST API
* 🧠 **LangChain ReAct Agent** — Custom tools for weather lookup, city comparison, and memory
* 🌐 **Google Gemini** — State-of-the-art LLM for natural conversations
* 🧱 **Redis via Docker** — Robust session & memory storage (run locally via Docker)
* 🌦️ **OpenWeatherMap API** — Reliable, real-time weather data
* 🔐 **CORS Support** — Seamless frontend integration

---

## 📋 API Endpoints

| Endpoint                     | Method | Description                             |
| ---------------------------- | ------ | --------------------------------------- |
| `/chat`                      | POST   | Chat with the weather agent             |
| `/weather/{city}`            | GET    | Get weather details for a city          |
| `/compare`                   | POST   | Compare weather between multiple cities |
| `/history/{session_id}`      | GET    | Retrieve past chat for a session        |
| `/clear/{session_id}`        | DELETE | Clear memory for a session              |
| `/session/{session_id}`      | DELETE | Fully delete a session                  |
| `/sessions`                  | GET    | List all active sessions                |
| `/session/{session_id}/info` | GET    | Get metadata for a session              |
| `/health`                    | GET    | Check if the API is running             |

---

## ⚙️ How It Works

* 🧠 **Session Management**: Unique session IDs allow Redis to persist conversations efficiently.
* ⟳ **Memory Recall**: Users enjoy context-aware chats with past interactions remembered.
* 🌍 **Weather Tools**: Weather fetch, city comparison, and history recall are all agent tools.
* 🤖 **LLM Intelligence**: Gemini answers general, non-weather queries like a true assistant.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/weather-agent-api.git
cd weather-agent-api
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API Keys

Add your keys to an `.env` or config file:

* **Google Gemini API Key**
* **OpenWeatherMap API Key**

### 4. Start Redis via Docker

Make sure Docker is installed and run:

```bash
docker run -d --name redis-weather -p 6379:6379 redis
```

### 5. Launch the API

```bash
uvicorn weather:app --reload
```

### 6. Open Swagger Docs

Navigate to: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💡 Example Interactions

### 🔹 Chat with the Agent

```http
POST /chat
Content-Type: application/json

{
  "query": "What's the weather in Paris?",
  "session_id": null
}
```

### 🔹 Compare Weather Across Cities

```http
POST /compare
Content-Type: application/json

{
  "cities": ["London", "New York", "Tokyo"]
}
```

---

## 💻 Frontend UI

A responsive **HTML interface** is included in the project to simulate chat sessions. It supports:

* 🔑 Custom session ID tracking
* 🧠 Memory recall from Redis
* 💬 Smooth user-agent interaction via FastAPI endpoints

Simply open the provided HTML file in a browser once the FastAPI app is running.

---

## 🛠️ Technologies Used

* **Python 3.11+**
* **FastAPI**
* **LangChain**
* **Google Gemini (Generative AI)**
* **Redis (Dockerized)**
* **OpenWeatherMap API**

---

## 🌟 Highlights

* ✅ **Production-Ready**: Graceful error handling, Redis TTL, and modular design
* ⚛️ **Extensible**: Easily add new tools, endpoints, or LLM providers
* 🔮 **Modern AI**: Combines LLM reasoning with real-world data
* 🛠️ **Frontend-Ready**: Built-in HTML UI, CORS-enabled

---

## 📅 License

This project is licensed under the **MIT License**.

---

## 👤 Author

* \[Goutham Nivass] (Sr Ai Engineer)

---
