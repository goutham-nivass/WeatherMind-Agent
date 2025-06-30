# import redis

# r = redis.Redis(host='localhost', port=6379)
# print(r.ping())  # Should print: True


import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

session_id = "be8d3ab0-ded7-4f6b-ab8d-e5be8151cf43"
data = r.get(f"weather_agent:session:{session_id}")  # Note the colon

if data:
    session_data = json.loads(data)
    messages = session_data['memory_data']['messages']
    print(f"Found {len(messages)} messages:")
    for msg in messages:
        print(f"[{msg['type'].upper()}]: {msg['content']}")
else:
    print("No data found.")





