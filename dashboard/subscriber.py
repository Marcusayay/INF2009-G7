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
DEFAULT_TOTALS = {
    "plastic": {"cans": 0, "bottles": 0},
    "metal": {"cans": 0, "bottles": 0},
    "glass": {"cans": 0, "bottles": 0},
    "general": {"cans": 0, "bottles": 0, "others": 0},
    "tetra": {"cartons": 0}
}

# Load existing data or start fresh
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        stored_data = json.load(f)
else:
    stored_data = {
        "totals": DEFAULT_TOTALS,
        "history": [] # <--- This is where we store the "Cache"
    }

# Backfill new materials/keys in older saved files
for material, default_counts in DEFAULT_TOTALS.items():
    stored_data.setdefault("totals", {}).setdefault(material, default_counts.copy())
    for key, value in default_counts.items():
        stored_data["totals"][material].setdefault(key, value)
stored_data.setdefault("history", [])

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
        if "bottle" in raw_type:
            key_type = "bottles"
        elif "carton" in raw_type:
            key_type = "cartons"
        elif "other" in raw_type:
            key_type = "others"
        elif "can" in raw_type:
            key_type = "cans"
        else:
            print(f"⏭️ Ignored unknown item type: {raw_type}")
            return

        # Plastic and glass cans are no longer tracked.
        if mat in ("plastic", "glass") and key_type == "cans":
            print(f"⏭️ Ignored unsupported type for {mat}: {raw_type}")
            return

        if mat not in stored_data["totals"]:
            if mat == "tetra":
                stored_data["totals"][mat] = {"cartons": 0}
            elif mat == "general":
                stored_data["totals"][mat] = {"cans": 0, "bottles": 0, "others": 0}
            else:
                stored_data["totals"][mat] = {"cans": 0, "bottles": 0}

        if key_type == "others" and mat != "general":
            print(f"⏭️ Ignored unsupported type for {mat}: {raw_type}")
            return

        if key_type not in stored_data["totals"][mat]:
            stored_data["totals"][mat][key_type] = 0

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