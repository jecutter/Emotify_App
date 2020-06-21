
# Emotify_App

This repository contains the code to run the Streamlit application for my Insight project: Emotify!

For full details on data used, model development, etc., see [here](https://github.com/jecutter/Emotify_Insight_Project).

This app was built to run on an AWS EC2 instance via [Streamlit](https://www.streamlit.io/), which is an extremely convenient pure-Python platform for loading and running ML models in a simple user interface.

## Requirements

Streamlit does require a recent version of Python (>= 3.6).

For this application specifically:
- python3.7
- pip install:
	* streamlit
	* numpy
	* pandas
	* scikit-learn
	* spotipy
	* nltk
	* langdetect
	* lyricsgenius
	* fuzzywuzzy
	* python-Levenshtein

This application also requires the following environment variables to be set:
- GENIUS_API_KEY: token for querying Genius API
- SPOTIPY_CLIENT_ID: Spotify client ID (from developer account)
- SPOTIPY_CLIENT_SECRET: Spotify client secret (from developer account)


