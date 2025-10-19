import os
os.environ["STREAMLIT_WATCHDOG_OBSERVER_TYPE"] = "polling"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import pickle
import pandas as pd

with open("model-reg-67130701919.pkl", "rb") as file:
    model = pickle.load(file)

new_data = pd.DataFrame({
    "youtube": [50],
    "tiktok": [50],
    "instagram": [50]
})

predicted_sales = model.predict(new_data)
print("Estimated sales:", predicted_sales[0])
