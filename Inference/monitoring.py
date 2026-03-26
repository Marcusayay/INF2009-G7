import psutil
import os
import re 
import subprocess 

def track_ram(): 
    return psutil.virtual_memory().percent

def track_temp(): 
    try:
        temps = psutil.sensors_temperatures()
        return temps["cpu_thermal"][0].current  # common on Raspberry Pi
    except Exception:
        return 0.0

#! the index foudn was 
#! VDD_CORE (CPU Power) -> current7, volt15 
#! 1V1_SYS (System Power) -> current5, volt13 
#! 0V8_SW (Switching regulator) -> current6, volt14 
#! 1V8_SYS (System Power) -> current2, volt10 
def track_power():
    try:
        res = subprocess.check_output(['vcgencmd', 'pmic_read_adc']).decode()
        
        # Helper to extract value by specific index
        def get_val(regex, index, data):
            match = re.search(rf'{regex}\({index}\)=([\d.]+)V?', data)
            return float(match.group(1)) if match else 0.0

        # Based on your output:
        # VDD_CORE (CPU Power)
        cpu_watt = get_val('current', 7, res) * get_val('volt', 15, res)
        
        # 1V1_SYS (System Power)
        sys_watt = get_val('current', 5, res) * get_val('volt', 13, res)
        
        # 0V8_SW (Switching regulator)
        sw_watt = get_val('current', 6, res) * get_val('volt', 14, res)

        # 1V8_SYS (System Power)
        sys_watt2 = get_val('current', 2, res) * get_val('volt', 10, res)

        return cpu_watt + sys_watt + sw_watt + sys_watt2 # Total estimated board power
    except Exception as e:
        print(f"Error: {e}")
        return 0.0

def returns_latest_file_number(directory : str):
    os.makedirs(directory, exist_ok=True)
    max_number = 0
    for filename in os.listdir(directory):
        # Regex to find digits immediately, optionally followed by an extension (handles both "file_1.jpg" and "file_1") 
        #match = re.search(r'(\d+)(?=\.[^.]+$)', filename)
        match = re.search(r'(\d+)(?:\.[^.]+)?$', filename)
        if match:
            # Convert to int immediately for comparison
            current_num = int(match.group(1))
            print (f"Found file: {filename} with number: {current_num}") 
            if current_num >= max_number:
                max_number = current_num
    return max_number + 1

if __name__ == "__main__": 
    print(f"RAM Usage: {track_ram()}%")
    print(f"CPU Temp: {track_temp()}°C")
    print(f"Estimated Power: {track_power():.2f}W") 
    print(returns_latest_file_number("output/mobilenet/quantized/run_1"))