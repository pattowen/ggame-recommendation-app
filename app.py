import streamlit as st
import pandas as pd
import numpy as np
import os

# Load data
data = pd.read_csv('recommend.csv')

# Define recommendation function
def recommend_games(selected_game, data, top_n=5):
    # Get the features of the selected game
    selected_game_data = data[data['game'] == selected_game]
    if selected_game_data.empty:
        return pd.DataFrame()  # Return empty if the selected game is not found
    
    # Extract the feature values for similarity
    selected_votes = selected_game_data['votes_up_count'].values[0]
    selected_playtime = selected_game_data['total_playtime'].values[0]
    
    # Compute similarity (Euclidean distance) for all other games
    data['similarity'] = np.sqrt(
        (data['votes_up_count'] - selected_votes) ** 2 +
        (data['total_playtime'] - selected_playtime) ** 2
    )
    
    # Exclude the selected game itself
    recommendations = data[data['game'] != selected_game]
    
    # Sort by similarity (ascending order)
    recommendations = recommendations.sort_values(by='similarity').head(top_n)
    return recommendations

# Streamlit app
st.title("Game Recommendation System")

# Game selection
selected_game = st.selectbox(
    "Select a game to get recommendations:",
    data['game'].unique(),
    key="game_select"
)

# Path to image folder (assuming images are in the 'image' folder)
image_folder = './image/'

# Recommendations Section
if st.button("Recommend Games"):
    with st.expander("Recommendations", expanded=True):
        recommended_games = recommend_games(selected_game, data)
        if recommended_games.empty:
            st.write("No recommendations available for this game.")
        else:
            st.write("**Recommended Games:**")
            rec_cols = st.columns(5)
            for rec_col, (_, game_row) in zip(rec_cols, recommended_games.iterrows()):
                rec_col.write(f"**{game_row['game']}**")  # Game name
                rec_col.write(f"Votes Up: {int(game_row['votes_up_count'])}")  # Votes up count
                rec_col.write(f"Total Playtime: {int(game_row['total_playtime'])} hours")  # Total playtime
                
                # Display the image of the recommended game
                recommended_image_path = os.path.join(image_folder, f"{game_row['game'][:3]}.jpg")
                if os.path.exists(recommended_image_path):
                    rec_col.image(recommended_image_path, caption=game_row['game'], use_container_width =True)
                else:
                    rec_col.write("Image not available")
