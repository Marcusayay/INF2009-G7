import qwiic_as7343
import serial
import time
import sys

# 1. Setup Serial
try:
    ser = serial.Serial('/tmp/ttyV0', 115200, timeout=1)
except:
    print("Serial Error: Ensure socat is running in another terminal!")
    sys.exit()

# 2. Initialize Sensor
sensor = qwiic_as7343.QwiicAS7343()
if not sensor.is_connected():
    print("AS7343 not found! Check I2C connections.")
    sys.exit()

sensor.begin()

# 3. Setup the 14-channel Mapping
# Even though it's 14 physical channels, the library often uses '18' 
# to ensure it captures every possible register.
sensor.set_auto_smux(sensor.kAutoSmux18Channels)
sensor.spectral_measurement_enable()

print("Streaming 14 Channels to Serial Plotter... Press Ctrl+C to stop.")

# 4. Correct Header for 14 Channels (F1-F8, NIR, Clear, and others)
# This names the lines in the SparkFun Plotter legend
header = "F1_405nm,F2_425nm,F3_450nm,F4_475nm,F5_515nm,F6_555nm,F7_600nm,F8_670nm,NIR,Clear,FD,IR1,IR2,IR3\n"
ser.write(header.encode('utf-8'))

try:
    while True:
        # Trigger the read
        sensor.read_all_spectral_data()
        
        # Grab exactly 14 channels from the internal data buffer
        data_list = sensor._data[:14] 
        
        if data_list and len(data_list) >= 14:
            csv_string = ",".join(map(str, data_list)) + "\n"
            ser.write(csv_string.encode('utf-8'))
            
            # Print to terminal so you see it's working
            print(f"Sent 14 Channels: {csv_string.strip()}")
        else:
            print("Data buffer empty... check sensor.")
            
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopping...")
    ser.close()