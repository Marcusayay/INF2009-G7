import paho.mqtt.client as mqtt
import json
import random
import time

# --- CONFIGURATION ---
BROKER = "10.127.71.107" # or your Pi's IP address
PORT = 1883
TOPIC = "pi/raw_transaction"

# --- SIMULATION DATA ---
materials = ["plastic", "metal", "glass", "general"]
item_types = ["cans", "bottles"] # matching your state keys

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

try:
    client.connect(BROKER, PORT, 60)
    print(f"Started simulation. Publishing to {TOPIC}...")

    while True:
        # 1. Generate random defaulted data
        payload = {
            "material": random.choice(materials),
            "type": random.choice(item_types),
            "weight": f"{random.randint(10, 250)}g"
        }

        # 2. Publish to the raw transaction topic
        client.publish(TOPIC, json.dumps(payload))
        
        print(f"🚀 Sent: {payload['material']} {payload['type']} ({payload['weight']})")
        
        # 3. Wait before next detection (e.g., 5 seconds)
        time.sleep(5)

except KeyboardInterrupt:
    print("\nSimulation stopped.")
    client.disconnect()
except Exception as e:
    print(f"Error: {e}")