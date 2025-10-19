# Step 1: Load the model
import pickle
import pandas as pd

with open("model-reg-67130701919.pkl", "rb") as file:
    model = pickle.load(file)

# Step 2: Create a new DataFrame
new_data = pd.DataFrame({
    "youtube": [50],
    "tiktok": [50],
    "instagram": [50]
})

# Step 3: Make predictions
predicted_sales = model.predict(new_data)

print("Estimated sales:", predicted_sales[0])
