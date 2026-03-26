import paho.mqtt.client as mqtt

# Define what happens when we connect to the broker
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected successfully!")
        # Subscribe to a topic (e.g., 'pi/sensors')
        client.subscribe("pi/test")
    else:
        print(f"Failed to connect, return code {reason_code}")

# Define what happens when a message is received
def on_message(client, userdata, msg):
    print(f"Received message on topic {msg.topic}: {msg.payload.decode()}")

# 1. Initialize the client using the 2.x API version
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# 2. Assign the callbacks
client.on_connect = on_connect
client.on_message = on_message

# 3. Connect to the local broker (localhost)
# If connecting from a DIFFERENT device, use the Pi's IP address here
client.connect("localhost", 1883, 60)

# 4. Start the loop to listen for messages forever
print("Subscriber started. Waiting for messages...")
client.loop_forever()