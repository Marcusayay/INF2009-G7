import os 


print ("📂 LISTING ALL MODELS IN .models FOLDER:" )
for file in os.listdir("."):
    print(file)
    if file.endswith(".py"): 
        print (f"type:  {type(file)}")
        print(f"✅ Found model file: {file}")