import os

# YOUR DATASET PATH
dataset_path = r"C:\VED\CTAI\Data"

print("🔍 Testing Dataset Structure...")
print(f"Checking: {dataset_path}")

if not os.path.exists(dataset_path):
    print("❌ FOLDER NOT FOUND!")
    print("Create folder: C:\\VED\\CTAI\\Data")
    print("Put your gesture folders inside it")
    exit()

# List all folders (gestures)
folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
folders.sort()

print(f"✅ Found {len(folders)} gesture folders:")
for i, folder in enumerate(folders[:20]):  # Show first 20
    folder_path = os.path.join(dataset_path, folder)
    images = [f for f in os.listdir(folder_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"   {i+1:2d}. {folder}: {len(images)} images")

print(f"\n🎉 Dataset READY! Total folders: {len(folders)}")