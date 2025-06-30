from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import requests
import json
import pickle
import redis
from datetime import datetime
import re
import uuid
from contextlib import asynccontextmanager

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage


REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None
REDIS_SESSION_TTL = 3600 * 24

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True
)


class WeatherRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class WeatherResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime

class WeatherComparisonRequest(BaseModel):
    cities: List[str]

class SessionRequest(BaseModel):
    session_id: str

class ConversationHistory(BaseModel):
    messages: List[Dict[str, Any]]
    session_id: str


class RedisSessionManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.session_prefix = "weather_agent:session:"
        self.session_ttl = REDIS_SESSION_TTL
    def save_session(self, session_id: str, agent: 'WeatherAgent'):
        try:
            memory_data = {
                'messages': [],
                'k': agent.memory.k,
                'return_messages': agent.memory.return_messages
            }
            for message in agent.memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    memory_data['messages'].append({
                        'type': 'human',
                        'content': message.content
                    })
                elif isinstance(message, AIMessage):
                    memory_data['messages'].append({
                        'type': 'ai',
                        'content': message.content
                    })
            session_data = {
                'memory_data': memory_data,
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat()
            }
            key = f"{self.session_prefix}{session_id}"
            self.redis.set(key, json.dumps(session_data, default=str), ex=self.session_ttl)
            print(f"✅ Session {session_id} saved to Redis with {len(memory_data['messages'])} messages")
            return True
        except Exception as e:
            print(f"Error saving session {session_id}: {e}")
            return False
    def load_session(self, session_id: str, gemini_api_key: str, weather_api_key: str) -> Optional['WeatherAgent']:
        try:
            key = f"{self.session_prefix}{session_id}"
            data = self.redis.get(key)
            if not data:
                return None
            session_data = json.loads(data)
            memory_data = session_data['memory_data']
            memory = ConversationBufferWindowMemory(
                k=memory_data['k'],
                return_messages=memory_data['return_messages'],
                memory_key="chat_history"
            )
            for msg in memory_data['messages']:
                if msg['type'] == 'human':
                    memory.chat_memory.add_user_message(msg['content'])
                elif msg['type'] == 'ai':
                    memory.chat_memory.add_ai_message(msg['content'])
            agent = WeatherAgent(gemini_api_key, weather_api_key, memory)
            session_data['last_accessed'] = datetime.now().isoformat()
            self.redis.set(key, pickle.dumps(session_data), ex=self.session_ttl)
            return agent
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None
    def delete_session(self, session_id: str) -> bool:
        try:
            key = f"{self.session_prefix}{session_id}"
            result = self.redis.delete(key)
            return result > 0
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            return False
    def session_exists(self, session_id: str) -> bool:
        try:
            key = f"{self.session_prefix}{session_id}"
            return self.redis.exists(key) > 0
        except Exception as e:
            print(f"Error checking session {session_id}: {e}")
            return False
    def list_sessions(self) -> List[str]:
        try:
            pattern = f"{self.session_prefix}*"
            keys = self.redis.keys(pattern)
            session_ids = [key.decode('utf-8').replace(self.session_prefix, '') for key in keys]
            return session_ids
        except Exception as e:
            print(f"Error listing sessions: {e}")
            return []
    def extend_session_ttl(self, session_id: str) -> bool:
        try:
            key = f"{self.session_prefix}{session_id}"
            return self.redis.expire(key, self.session_ttl)
        except Exception as e:
            print(f"Error extending TTL for session {session_id}: {e}")
            return False
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            key = f"{self.session_prefix}{session_id}"
            data = self.redis.get(key)
            if not data:
                return None
            session_data = pickle.loads(data)
            ttl = self.redis.ttl(key)
            return {
                'session_id': session_id,
                'created_at': session_data['created_at'],
                'last_accessed': session_data['last_accessed'],
                'ttl_seconds': ttl,
                'message_count': len(session_data['memory_data']['messages'])
            }
        except Exception as e:
            print(f"Error getting session info {session_id}: {e}")
            return None

class WeatherService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    def get_weather(self, city: str) -> Optional[Dict[str, Any]]:
        try:
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                weather_info = {
                    'city': data['name'],
                    'country': data['sys']['country'],
                    'temperature': data['main']['temp'],
                    'feels_like': data['main']['feels_like'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'description': data['weather'][0]['description'],
                    'wind_speed': data['wind']['speed'],
                    'visibility': data.get('visibility', 'N/A'),
                    'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M'),
                    'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M')
                }
                return weather_info
            else:
                return None
        except Exception as e:
            print(f"Weather API Error: {e}")
            return None

class WeatherAgent:
    def __init__(self, gemini_api_key: str, weather_api_key: str, memory: Optional[ConversationBufferWindowMemory] = None):
        self.gemini_api_key = gemini_api_key
        self.weather_api_key = weather_api_key
        self.weather_service = WeatherService(weather_api_key)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_api_key,
            temperature=0.7,
            max_tokens=1024
        )
        self.memory = memory or ConversationBufferWindowMemory(
            k=50,
            return_messages=True,
            memory_key="chat_history"
        )
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()
    def _get_ordinal(self, n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"
    def _create_tools(self) -> list:
        def general_knowledge_tool(query: str) -> str:
            try:
                response = self.llm.invoke(f"Please answer this question: {query}")
                return response.content
            except Exception as e:
                return f"I encountered an error processing your question: {str(e)}"
        def get_weather_tool(city: str) -> str:
            weather_data = self.weather_service.get_weather(city)
            if weather_data:
                return self._format_weather_response(weather_data)
            else:
                return f"Sorry, I couldn't find weather information for '{city}'. Please check the city name and try again."
        def get_weather_comparison_tool(cities: str) -> str:
            city_list = [city.strip() for city in cities.split(',')]
            if len(city_list) < 2:
                return "Please provide at least 2 cities separated by commas for comparison."
            weather_data = []
            for city in city_list[:5]:
                data = self.weather_service.get_weather(city)
                if data:
                    weather_data.append(data)
            if not weather_data:
                return "Sorry, I couldn't fetch weather data for any of the provided cities."
            return self._format_weather_comparison(weather_data)
        def recall_conversation_tool(query: str) -> str:
            import re
            chat_history = self.memory.chat_memory.messages
            questions = [msg.content for msg in chat_history if isinstance(msg, HumanMessage)]
            query_lower = query.lower()
            number_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s+question', query_lower)
            if number_match:
                question_num = int(number_match.group(1))
                if question_num <= len(questions) and question_num > 0:
                    ordinal = self._get_ordinal(question_num)
                    return f"Your {ordinal} question was: '{questions[question_num - 1]}'"
                else:
                    return f"You haven't asked {question_num} questions yet. You've asked {len(questions)} questions so far."
            if "first question" in query_lower:
                if questions:
                    return f"Your first question was: '{questions[0]}'"
                else:
                    return "You haven't asked any questions yet."
            if "last question" in query_lower or "previous question" in query_lower:
                if len(questions) >= 2:
                    return f"Your previous question was: '{questions[-2]}'"
                elif len(questions) == 1:
                    return f"This is only your second interaction. Your first question was: '{questions[0]}'"
                else:
                    return "You haven't asked any previous questions yet."
            if "question count" in query_lower or "how many questions" in query_lower:
                return f"You have asked {len(questions)} questions so far in our conversation."
            if "all questions" in query_lower or "list questions" in query_lower:
                if questions:
                    question_list = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
                    return f"Here are all your questions:\n{question_list}"
                else:
                    return "You haven't asked any questions yet."
            available_options = [
                "specific question number (e.g., 'second question', 'third question')",
                "first question",
                "last/previous question", 
                "question count",
                "all questions"
            ]
            return f"I can help you recall: {', '.join(available_options)}. You've asked {len(questions)} questions so far."
        tools = [
            Tool(
                name="general_knowledge",
                func=general_knowledge_tool,
                description="Answer general knowledge questions about any topic including machine learning, science, technology, history, etc. Use this for non-weather related questions."
            ),
            Tool(
                name="get_weather",
                func=get_weather_tool,
                description="Get current weather information for a specific city. Input should be the city name."
            ),
            Tool(
                name="compare_weather",
                func=get_weather_comparison_tool,
                description="Compare weather between multiple cities. Input should be city names separated by commas."
            ),
            Tool(
                name="recall_conversation",
                func=recall_conversation_tool,
                description="Recall information from previous conversation. Can retrieve: specific question by number (e.g. 'second question', 'third question'), first question, last/previous question, question count, or list all questions."
            )
        ]
        return tools
    def _create_agent(self) -> AgentExecutor:
        template = """You are a helpful AI assistant with access to weather data and general knowledge. Answer questions as best you can using the available tools.

    Available tools:
    {tools}

    Use the following guidelines:
    - For weather-related questions (current weather, forecasts, comparisons), use the weather tools
    - For general knowledge questions (science, technology, history, machine learning, etc.), use the general_knowledge tool
    - For conversation history questions, use the recall_conversation tool
    - Always think step by step about which tool is most appropriate for the question

    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do and which tool to use
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Previous conversation:
    {chat_history}

    Question: {input}
    Thought: {agent_scratchpad}"""
        prompt = PromptTemplate(
            input_variables=["input", "chat_history", "agent_scratchpad"],
            template=template
        )
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        return agent_executor
    def _format_weather_response(self, weather_data: Dict[str, Any]) -> str:
        visibility_km = weather_data['visibility']/1000 if weather_data['visibility'] != 'N/A' else 'N/A'
        response = f"""🌤️ **Weather in {weather_data['city']}, {weather_data['country']}**

🌡️ **Temperature:** {weather_data['temperature']:.1f}°C (feels like {weather_data['feels_like']:.1f}°C)
📝 **Condition:** {weather_data['description'].title()}
💧 **Humidity:** {weather_data['humidity']}%
🎈 **Pressure:** {weather_data['pressure']} hPa
💨 **Wind Speed:** {weather_data['wind_speed']} m/s
👁️ **Visibility:** {visibility_km} km
🌅 **Sunrise:** {weather_data['sunrise']}
🌇 **Sunset:** {weather_data['sunset']}\n\n*Weather data provided by OpenWeatherMap*"""
        return response
    def _format_weather_comparison(self, weather_data_list: list) -> str:
        if not weather_data_list:
            return "No weather data available for comparison."
        response = "🌍 **Weather Comparison**\n\n"
        for data in weather_data_list:
            response += f"📍 **{data['city']}, {data['country']}**\n"
            response += f"   🌡️ {data['temperature']:.1f}°C ({data['description'].title()})\n"
            response += f"   💧 Humidity: {data['humidity']}% | 💨 Wind: {data['wind_speed']} m/s\n\n"
        temps = [data['temperature'] for data in weather_data_list]
        hottest = weather_data_list[temps.index(max(temps))]
        coldest = weather_data_list[temps.index(min(temps))]
        response += f"📊 **Summary:**\n"
        response += f"🔥 Hottest: {hottest['city']} ({max(temps):.1f}°C)\n"
        response += f"❄️ Coldest: {coldest['city']} ({min(temps):.1f}°C)\n"
        return response
    def _recall_first_question(self) -> str:
        chat_history = self.memory.chat_memory.messages
        for message in chat_history:
            if isinstance(message, HumanMessage):
                return f"The first question you asked was: '{message.content}'"
        return "This is actually your first question! I don't have any previous questions in memory yet."
    def _get_question_count(self) -> str:
        chat_history = self.memory.chat_memory.messages
        question_count = len([msg for msg in chat_history if isinstance(msg, HumanMessage)])
        return f"You have asked {question_count} questions so far in our conversation."
    def process_query(self, user_input: str) -> str:
        try:
            if self._is_simple_weather_query(user_input):
                city = self._extract_city_from_query(user_input)
                if city:
                    weather_data = self.weather_service.get_weather(city)
                    if weather_data:
                        response = self._format_weather_response(weather_data)
                        self.memory.chat_memory.add_user_message(user_input)
                        self.memory.chat_memory.add_ai_message(response)
                        return response
            response = self.agent_executor.invoke({"input": user_input})
            return response.get("output", "I'm sorry, I couldn't process your request.")
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    def _is_simple_weather_query(self, query: str) -> bool:
        weather_keywords = ['weather', 'temperature', 'climate']
        location_patterns = [r'weather in (.+)', r'temperature in (.+)', r'weather for (.+)']
        query_lower = query.lower()
        has_weather_keyword = any(keyword in query_lower for keyword in weather_keywords)
        has_location_pattern = any(re.search(pattern, query_lower) for pattern in location_patterns)
        return has_weather_keyword and has_location_pattern
    def _extract_city_from_query(self, query: str) -> Optional[str]:
        patterns = [
            r"weather in (.+?)(?:\?|$)",
            r"weather for (.+?)(?:\?|$)",
            r"temperature in (.+?)(?:\?|$)",
            r"climate in (.+?)(?:\?|$)"
        ]
        query_lower = query.lower()
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                city = match.group(1).strip()
                city = re.sub(r'\b(today|now|currently|right now)\b', '', city).strip()
                return city if city else None
        return None
    def get_conversation_history(self) -> list:
        history = []
        chat_messages = self.memory.chat_memory.messages
        for message in chat_messages:
            if isinstance(message, HumanMessage):
                history.append({
                    'type': 'human',
                    'content': message.content,
                    'timestamp': datetime.now().isoformat()
                })
            elif isinstance(message, AIMessage):
                history.append({
                    'type': 'ai',
                    'content': message.content,
                    'timestamp': datetime.now().isoformat()
                })
        return history
    def clear_memory(self):
        self.memory.clear()


GEMINI_API_KEY = ""
WEATHER_API_KEY = ""
session_manager = RedisSessionManager(redis_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Weather Agent API with Redis...")
    try:
        redis_client.ping()
        print("✅ Redis connection established")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Please make sure Redis is running")
    yield
    print("Shutting down Weather Agent API...")
    redis_client.close()


app = FastAPI(
    title="Weather Agent API with Redis",
    description="A LangChain-powered weather agent with Redis session management",
    version="2.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, WeatherAgent]:
    if session_id:
        agent = session_manager.load_session(session_id, GEMINI_API_KEY, WEATHER_API_KEY)
        if agent:
            session_manager.extend_session_ttl(session_id)
            return session_id, agent
    new_session_id = str(uuid.uuid4())
    agent = WeatherAgent(GEMINI_API_KEY, WEATHER_API_KEY)
    session_manager.save_session(new_session_id, agent)
    return new_session_id, agent
@app.post("/chat", response_model=WeatherResponse)
async def chat_with_agent(request: WeatherRequest):
    try:
        session_id, agent = get_or_create_session(request.session_id)
        response = agent.process_query(request.query)
        save_success = session_manager.save_session(session_id, agent)
        print(f"Save result for session {session_id}: {save_success}")
        return WeatherResponse(
            response=response,
            session_id=session_id,
            timestamp=datetime.now()
        )
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
@app.get("/")
async def root():
    try:
        redis_status = "connected" if redis_client.ping() else "disconnected"
        active_sessions = len(session_manager.list_sessions())
    except Exception:
        redis_status = "error"
        active_sessions = 0
    return {
        "message": "Weather Agent API with Redis",
        "version": "2.0.0",
        "redis_status": redis_status,
        "active_sessions": active_sessions,
        "endpoints": {
            "/chat": "POST - Chat with the weather agent",
            "/weather/{city}": "GET - Get weather for a specific city",
            "/compare": "POST - Compare weather between cities",
            "/history/{session_id}": "GET - Get conversation history",
            "/clear/{session_id}": "DELETE - Clear session memory",
            "/session/{session_id}": "DELETE - Delete session completely",
            "/sessions": "GET - List all active sessions",
            "/session/{session_id}/info": "GET - Get session metadata",
            "/health": "GET - Health check"
        }
    }
@app.post("/chat", response_model=WeatherResponse)
async def chat_with_agent(request: WeatherRequest):
    try:
        session_id, agent = get_or_create_session(request.session_id)
        response = agent.process_query(request.query)
        session_manager.save_session(session_id, agent)
        return WeatherResponse(
            response=response,
            session_id=session_id,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
@app.get("/debug/redis/{session_id}")
async def debug_redis_session(session_id: str):
    try:
        key = f"{session_manager.session_prefix}{session_id}"
        data = redis_client.get(key)
        if not data:
            return {"error": "Session not found in Redis"}
        session_data = pickle.loads(data)
        return {
            "session_exists": True,
            "message_count": len(session_data['memory_data']['messages']),
            "messages": session_data['memory_data']['messages'],
            "created_at": session_data['created_at'],
            "last_accessed": session_data['last_accessed']
        }
    except Exception as e:
        return {"error": str(e)}
@app.get("/weather/{city}")
async def get_weather(city: str, session_id: Optional[str] = None):
    try:
        weather_service = WeatherService(WEATHER_API_KEY)
        weather_data = weather_service.get_weather(city)
        if weather_data:
            result_session_id = session_id
            if session_id:
                agent = session_manager.load_session(session_id, GEMINI_API_KEY, WEATHER_API_KEY)
                if not agent:
                    result_session_id = str(uuid.uuid4())
                    agent = WeatherAgent(GEMINI_API_KEY, WEATHER_API_KEY)
                query = f"weather in {city}"
                response = agent._format_weather_response(weather_data)
                agent.memory.chat_memory.add_user_message(query)
                agent.memory.chat_memory.add_ai_message(response)
                session_manager.save_session(result_session_id, agent)
            return {
                "success": True,
                "data": weather_data,
                "session_id": result_session_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Weather data not found for city: {city}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching weather data: {str(e)}")
@app.post("/compare")
async def compare_weather(request: WeatherComparisonRequest, session_id: Optional[str] = None):
    try:
        weather_service = WeatherService(WEATHER_API_KEY)
        weather_data = []
        for city in request.cities[:5]:
            data = weather_service.get_weather(city)
            if data:
                weather_data.append(data)
        if not weather_data:
            raise HTTPException(status_code=404, detail="No weather data found for the provided cities")
        result_session_id = session_id
        if session_id:
            agent = session_manager.load_session(session_id, GEMINI_API_KEY, WEATHER_API_KEY)
            if not agent:
                result_session_id = str(uuid.uuid4())
                agent = WeatherAgent(GEMINI_API_KEY, WEATHER_API_KEY)
            query = f"compare weather between {', '.join(request.cities)}"
            response = agent._format_weather_comparison(weather_data)
            agent.memory.chat_memory.add_user_message(query)
            agent.memory.chat_memory.add_ai_message(response)
            session_manager.save_session(result_session_id, agent)
        return {
            "success": True,
            "data": weather_data,
            "comparison_count": len(weather_data),
            "session_id": result_session_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing weather data: {str(e)}")
@app.get("/history/{session_id}", response_model=ConversationHistory)
async def get_conversation_history(session_id: str):
    try:
        agent = session_manager.load_session(session_id, GEMINI_API_KEY, WEATHER_API_KEY) 
        if not agent:
            raise HTTPException(status_code=404, detail="Session not found")
        history = agent.get_conversation_history()
        session_manager.extend_session_ttl(session_id)
        return ConversationHistory(
            messages=history,
            session_id=session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")
@app.delete("/clear/{session_id}")
async def clear_session_memory(session_id: str):
    try:
        agent = session_manager.load_session(session_id, GEMINI_API_KEY, WEATHER_API_KEY)
        if not agent:
            raise HTTPException(status_code=404, detail="Session not found")
        agent.clear_memory()
        session_manager.save_session(session_id, agent)
        return {
            "success": True,
            "message": f"Memory cleared for session {session_id}",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")
@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    try:
        success = session_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "success": True,
            "message": f"Session {session_id} deleted successfully",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")
@app.get("/sessions")
async def list_sessions():
    try:
        active_sessions = session_manager.list_sessions()
        return {
            "active_sessions": active_sessions,
            "session_count": len(active_sessions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")
@app.get("/session/{session_id}/info")
async def get_session_info(session_id: str):
    try:
        info = session_manager.get_session_info(session_id)
        if not info:
            raise HTTPException(status_code=404, detail="Session not found")
        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session info: {str(e)}")
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        redis_status = "healthy" if redis_client.ping() else "unhealthy"
        active_sessions = len(session_manager.list_sessions())
    except Exception as e:
        redis_status = f"error: {str(e)}"
        active_sessions = 0
    return {
        "status": "healthy",
        "redis_status": redis_status,
        "timestamp": datetime.now().isoformat(),
        "active_sessions": active_sessions
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)