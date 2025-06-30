# WeatherMind Agent: A FastAPI, Redis, and LangChain-Powered Conversational Weather Assistant

## Overview

Weather Agent API is a robust, production-ready FastAPI application that leverages LangChain, Google Gemini, and Redis to provide:
- Real-time weather information for any city
- Weather comparison between multiple cities
- Persistent, stateful chat sessions with conversation memory
- General knowledge Q&A via LLM

This project is designed for extensibility, reliability, and modern AI-powered user experiences.

---

## Features

- **FastAPI Backend**: High-performance, async REST API
- **LangChain Agent**: ReAct agent with custom tools for weather, comparison, recall, and general knowledge
- **Google Gemini LLM**: Advanced language model for natural conversation
- **Redis Session Management**: Scalable, persistent session and memory storage
- **OpenWeatherMap Integration**: Accurate, up-to-date weather data
- **CORS Enabled**: Ready for frontend integration

---

## Endpoints

| Endpoint                        | Method | Description                                      |
|---------------------------------|--------|--------------------------------------------------|
| `/chat`                         | POST   | Chat with the weather agent                      |
| `/weather/{city}`               | GET    | Get weather for a specific city                  |
| `/compare`                      | POST   | Compare weather between multiple cities          |
| `/history/{session_id}`         | GET    | Get conversation history for a session           |
| `/clear/{session_id}`           | DELETE | Clear conversation memory for a session          |
| `/session/{session_id}`         | DELETE | Delete a session completely                      |
| `/sessions`                     | GET    | List all active sessions                         |
| `/session/{session_id}/info`    | GET    | Get session metadata                             |
| `/health`                       | GET    | Health check                                     |

---

## How It Works

- **Session Management**: Each chat is associated with a unique session ID, stored in Redis for persistence and scalability.
- **Conversation Memory**: The agent remembers previous questions and answers, enabling context-aware responses.
- **Weather Tools**: Custom tools fetch and format weather data, compare cities, and recall conversation history.
- **General Knowledge**: Non-weather queries are answered by the Gemini LLM.

---

## Quick Start

1. **Clone the repository**
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set your API keys**
   - Google Gemini API Key
   - OpenWeatherMap API Key
4. **Start Redis server**
5. **Run the API**
   ```sh
   uvicorn weather:app --reload
   ```
6. **Test endpoints** using Swagger UI at `http://localhost:8000/docs`

---

## Example Usage

### Chat with the Agent
```json
POST /chat
{
  "query": "What's the weather in Paris?",
  "session_id": null
}
```

### Compare Weather
```json
POST /compare
{
  "cities": ["London", "New York", "Tokyo"]
}
```

---

## Technologies Used
- Python 3.11+
- FastAPI
- LangChain
- Google Gemini (Generative AI)
- Redis
- OpenWeatherMap API

---

## Professional Highlights
- **Production-Ready**: Robust error handling, session TTL, and scalable architecture
- **Extensible**: Easily add new tools, endpoints, or LLM providers
- **Modern AI**: Combines LLM reasoning with real-world data
- **Frontend-Ready**: CORS enabled for seamless integration

---

## License

This project is licensed under the MIT License.

---

## Authors

- [Your Name] (Project Lead)
- [Contributors]

---

## Contact

For support or business inquiries, please contact [your-email@example.com].
