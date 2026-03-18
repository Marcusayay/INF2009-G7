import tensorflow as tf

print("===========================================")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("All Detected Devices: ")
for device in tf.config.list_physical_devices():
    print(f" - {device.device_type}: {device.name}")
print("===========================================")