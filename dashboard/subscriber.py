# import paho.mqtt.client as mqtt
# import json
# import time

# # --- CONFIGURATION ---
# BROKER = "localhost"
# PORT = 1883
# # Dictionary to simulate/store our bin counts
# counts = {
#     "plastic": {"cans": 0, "bottles": 0},
#     "metal": {"cans": 0, "bottles": 0},
#     "glass": {"cans": 0, "bottles": 0},
#     "general": {"cans": 0, "bottles": 0}
# }

# def on_connect(client, userdata, flags, reason_code, properties):
#     if reason_code == 0:
#         print(f"✅ Connected to Broker at {BROKER}")
#         # Subscribe to the test topic to listen for manual triggers
#         client.subscribe("pi/test")
#     else:
#         print(f"❌ Connection failed: {reason_code}")

# def on_message(client, userdata, msg):
#     payload = msg.payload.decode()
#     print(f"📩 Received: {msg.topic} -> {payload}")
    
#     # Example logic: If we receive "plastic_bottle", increment and publish update
#     if payload == "plastic_bottle":
#         counts["plastic"]["bottles"] += 1
#         publish_update(client, "plastic")

# def publish_update(client, material):
#     """Sends the updated JSON to the specific material channel"""
#     topic = f"pi/material/{material}"
#     payload = json.dumps(counts[material])
#     client.publish(topic, payload)
#     print(f"📤 Published to {topic}: {payload}")

# # 1. Initialize Client
# client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
# client.on_connect = on_connect
# client.on_message = on_message

# # 2. Connect
# try:
#     client.connect(BROKER, PORT, 60)
# except Exception as e:
#     print(f"Could not connect: {e}")
#     exit(1)

# # 3. Start Loop
# print("🚀 Subscriber Logic Started...")
# print("Tip: Use another terminal to trigger an update:")
# print("mosquitto_pub -t 'pi/test' -m 'plastic_bottle'")

# client.loop_forever()

import paho.mqtt.client as mqtt
import json
import os
from datetime import datetime

DATA_FILE = "bin_stats.json"

# Load existing data or start fresh
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        stored_data = json.load(f)
else:
    stored_data = {
        "totals": {
            "plastic": {"cans": 0, "bottles": 0},
            "metal": {"cans": 0, "bottles": 0},
            "glass": {"cans": 0, "bottles": 0},
            "general": {"cans": 0, "bottles": 0}
        },
        "history": [] # <--- This is where we store the "Cache"
    }

def save_data():
    with open(DATA_FILE, "w") as f:
        json.dump(stored_data, f)

def on_message(client, userdata, msg):
    try:
        # Expected from Mac: {"material": "plastic", "type": "bottle", "weight": "15g"}
        tx = json.loads(msg.payload.decode())
        mat = tx["material"].lower()
        # Handle singular/plural consistency
        raw_type = tx["type"].lower()
        key_type = "bottles" if "bottle" in raw_type else "cans"

        # 1. Update Totals
        stored_data["totals"][mat][key_type] += 1

        # 2. Create Transaction Record
        new_entry = {
            "id": int(datetime.now().timestamp() * 1000),
            "material": mat.capitalize(),
            "type": raw_type.capitalize(),
            "weight": tx["weight"],
            "timestamp": datetime.now().strftime("%d/%m/%Y | %H:%M")
        }

        # 3. Add to History (Keep only the last 20 items)
        stored_data["history"].insert(0, new_entry)
        stored_data["history"] = stored_data["history"][:20]

        save_data()

        # 4. Publish to React (Send EVERYTHING)
        # React will use 'totals' for cards and 'history' for the scrolling list
        client.publish(f"pi/material/{mat}", json.dumps({
            "totals": stored_data["totals"][mat],
            "history": stored_data["history"]
        }))

        print(f"✅ Saved & Published update for {mat}")

    except Exception as e:
        print(f"❌ Error: {e}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = lambda c, u, f, r, p: c.subscribe("pi/raw_transaction")
client.on_message = on_message

client.connect("localhost", 1883)

print("🚀 Subscriber Logic Started...")

# Optional: Broadcast current state to all topics on startup
for mat in stored_data["totals"]:
    client.publish(f"pi/material/{mat}", json.dumps({
        "totals": stored_data["totals"][mat],
        "history": stored_data["history"]
    }))
    
client.loop_forever()