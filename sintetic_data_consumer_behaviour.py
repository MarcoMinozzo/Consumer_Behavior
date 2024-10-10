import pandas as pd
import numpy as np

# Constants
num_rows = 1000000

# Generate SESSION_ID as unique identifiers
session_ids = np.arange(1, num_rows + 1)

# Simulate user behavior with logical dependencies
# The probability of each action is linked to how much information the user has interacted with.
click_image = np.random.binomial(1, 0.2, num_rows)
read_review = np.random.binomial(1, 0.3, num_rows)
category_view = np.random.binomial(1, 0.5, num_rows)
read_details = np.random.binomial(1, 0.4, num_rows)
video_view = np.random.binomial(1, 0.25, num_rows)
add_to_list = np.random.binomial(1, 0.2, num_rows)
compare_prc = np.random.binomial(1, 0.15, num_rows)
view_similar = np.random.binomial(1, 0.1, num_rows)
save_for_later = np.random.binomial(1, 0.1, num_rows)
personalized = np.random.binomial(1, 0.05, num_rows)

# Create BUY logic based on increased engagement leading to more purchases
buy = (click_image + read_review + category_view + read_details + video_view + 
       add_to_list + compare_prc + view_similar + personalized) > 2

# Some users purchase without much engagement (random effect)
random_buyers = np.random.binomial(1, 0.05, num_rows)
buy = np.logical_or(buy, random_buyers).astype(int)

# Create DataFrame
data = {
    "SESSION_ID": session_ids,
    "Click_Image": click_image,
    "Read_Review": read_review,
    "Category_View": category_view,
    "Read_Details": read_details,
    "Video_View": video_view,
    "Add_to_List": add_to_list,
    "Compare_Prc": compare_prc,
    "View_Similar": view_similar,
    "Save_for_Later": save_for_later,
    "Personalized": personalized,
    "BUY": buy
}

df = pd.DataFrame(data)

# Save to CSV
file_path = "/mnt/data/consumer_behavior.csv"
df.to_csv(file_path, index=False)

file_path
