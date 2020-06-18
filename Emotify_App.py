import streamlit as st
import time
import os 
import sys 
import re 
import numpy as np
import pandas as pd
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from joblib import load
import lyricsgenius
from langdetect import detect
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from fuzzywuzzy import fuzz


'''

# Emotify

### *Creating an emotional backdrop to meet your needs - from any playlist*

'''
### *Finding songs to meet your emotional needs*


#####
#@st.cache
#def split_by_emotion(features, labels):
#    x_train1, x_test1, y_train1, y_test1 = train_test_split(features[labels.overall_emotion=='happy'], labels[labels.overall_emotion=='happy'], test_size=0.2, random_state=1)
#    x_train2, x_test2, y_train2, y_test2 = train_test_split(features[labels.overall_emotion=='calm'], labels[labels.overall_emotion=='calm'], test_size=0.2, random_state=2)
#    x_train3, x_test3, y_train3, y_test3 = train_test_split(features[labels.overall_emotion=='sad'], labels[labels.overall_emotion=='sad'], test_size=0.2, random_state=3)
#    x_train4, x_test4, y_train4, y_test4 = train_test_split(features[labels.overall_emotion=='angry'], labels[labels.overall_emotion=='angry'], test_size=0.2, random_state=4)
#
#    x_train = pd.concat([x_train1, x_train2, x_train3, x_train4])
#    x_test = pd.concat([x_test1, x_test2, x_test3, x_test4])
#    y_train = pd.concat([y_train1, y_train2, y_train3, y_train4])
#    y_test = pd.concat([y_test1, y_test2, y_test3, y_test4])
#
#    return x_train, x_test, y_train, y_test

##@st.cache
#def fit_rfc(suppress_st_warning=True, allow_output_mutation=True):
#	df_ds = pd.read_csv('Total_Spotify_4Emotion_AllParts.csv')
#	df_ds['loudness_linear'] = 10**(df_ds['loudness']/20.)
#	features = df_ds[['danceability', 'energy', 'key', 'loudness_linear',
#							'speechiness', 'acousticness', 'instrumentalness', 'liveness',
#							'valence', 'tempo', 'duration_ms']]
#	labels = df_ds[['overall_emotion']]
#	x_train, x_test, y_train, y_test = split_by_emotion(features, labels)
#	scale_train = StandardScaler().fit(x_train)
#	train_features = scale_train.transform(x_train)
#	pretrained_rfc = RandomForestClassifier(
#			n_estimators=2000,
#			max_depth=15,
#			min_samples_split=2,
#			min_samples_leaf=1,
#			random_state = 1)
#	pretrained_rfc.fit(train_features, y_train.values.ravel())
#	return pretrained_rfc, scale_train
#
#pretrained_rfc, scale_train = fit_rfc()
#####

#df_ds = pd.read_csv('Total_Spotify_4Emotion_AllParts.csv')
#df_ds['loudness_linear'] = 10**(df_ds['loudness']/20.)
#features = df_ds[['danceability', 'energy', 'key', 'loudness_linear',
#						'speechiness', 'acousticness', 'instrumentalness', 'liveness',
#						'valence', 'tempo', 'duration_ms']]
#labels = df_ds[['overall_emotion']]
#x_train, x_test, y_train, y_test = split_by_emotion(features, labels)
#scale_train = StandardScaler().fit(x_train)


# Create and embed function for embedding Spotify
# track player into the application
def embed_spotify_track(track_id):
	st.write(
					f'<iframe src="https://open.spotify.com/embed/track/{track_id}" width="450" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>',
					unsafe_allow_html=True,
			)

@st.cache(allow_output_mutation=True)
def load_models():
	# Load our pretrained models
	pretrained_inst_scale = pickle.load(open('models_v2/best_inst_scale.pkl', 'rb'))
	pretrained_inst_rfc = pickle.load(open('models_v2/best_inst_rfc.pkl', 'rb'))
	pretrained_lyrical_scale = pickle.load(open('models_v2/best_lyrical_scale.pkl', 'rb'))
	pretrained_lyrical_rfc = pickle.load(open('models_v2/best_lyrical_rfc.pkl', 'rb'))
	return pretrained_inst_scale, pretrained_inst_rfc, pretrained_lyrical_scale, pretrained_lyrical_rfc

@st.cache(allow_output_mutation=True)
def get_genius_api():
	# Authenticate Genius instance (for lyrics)
	GENIUS_API_KEY = os.getenv("GENIUS_API_KEY")
	genius = lyricsgenius.Genius(GENIUS_API_KEY)
	genius.verbose = False # turn off status messages 
	genius.remove_section_headers = True # turn off lyrics section headers
	return genius

#@st.cache(allow_output_mutation=True)
def get_spotify_api():
	# Authenticate Spotify instance (for audio)
	SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
	SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
	token = spotipy.oauth2.SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
	cache_token = token.get_access_token()
	sp = spotipy.Spotify(cache_token)
	return sp


# Load models and get API instances
pretrained_inst_scale, pretrained_inst_rfc, pretrained_lyrical_scale, pretrained_lyrical_rfc = load_models()
genius = get_genius_api()
sp = get_spotify_api()


# Specify a default playlist ID
#discover_uri = 'spotify:playlist:37i9dQZEVXcPBbdEIfyjYg'
#discover_id = '37i9dQZEVXcPBbdEIfyjYg'
discover_id = '37i9dQZF1DWTLSN7iG21yC'  # "Work From Home" playlist
discover_uri = 'spotify:playlist:'+discover_id
discover_link = 'https://open.spotify.com/playlist/'+discover_id

# Set a max wait time for Spotify querying and song classification
max_wait_time = 60 # sec


# Begin placing text and user interactions
st.text('\n')
st.text('\n')

# Placeholder if we want to replace an option menu with output
#placeholder = st.empty()

playlist_uri = st.text_input("Give a valid playlist ID/URI (copied from Spotify):", discover_id)

option_inst = st.selectbox('Do you want instrumental or lyrical music?',
													('Lyrical/Vocal', 'Instrumental'))

# Provide a selection box for choosing an emotion
option = st.selectbox('Select an emotion: ',
										 ('-', 'Happy', 'Sad', 'Angry'))
if option_inst == 'Instrumental':
	features_list = ['danceability', 'energy', 'loudness', 'speechiness',
       'acousticness', 'valence', 'duration_ms']
elif option_inst == 'Lyrical/Vocal':
	features_list = ['danceability', 'energy', 'key', 'loudness',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'duration_ms', 'mode', 'time_signature']
	
#features_list = ['danceability', 'energy', 'key', 'loudness', 
#						'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
#						'valence', 'tempo', 'duration_ms']


# Very important -- which emojis to use for which emotions!
#emoji_dict = {'Happy':':smile:', 'Calm':':relieved:', 'Sad':':pensive:', 'Angry':':angry:'}
emoji_dict = {'Happy':':smile:', 'Sad':':pensive:', 'Angry':':angry:'}


# Set variables needed in song loop
#characters = 'abcdefghijklmnopqrstuvwxyz'
songs = []

# Check for valid emotion selection before doing Spotify queries
if option == 'Happy' or option == 'Calm' or option == 'Sad' or option == 'Angry':
	with st.spinner('Querying Spotify and classifying songs for you...'):
		counter = 0
		#result = sp.user_playlist_tracks('spotify', discover_id)['items']
		try:
			result = sp.user_playlist_tracks('spotify', playlist_uri.rsplit(':', 1)[-1])['items']
		except:
			st.write('Could not find Spotify playlist. Using default playlist...')
			result = sp.user_playlist_tracks('spotify', discover_uri.rsplit(':', 1)[-1])['items']
	
		program_start = time.time()
		timed_out = False
		for track in result: 
			# Get metadata 
			artist = track["track"]["album"]["artists"][0]["name"] 
			album = track["track"]["album"]["name"] 
			track_name = track["track"]["name"] 
			track_id = track["track"]["id"] 
					 
			# Get audio features
			features = []
			audio_features = sp.audio_features(track_id)[0] 
			for feature in features_list:
				if feature == 'loudness':
					features.append(10**(audio_features[feature]/20.)) 
				else:
					features.append(audio_features[feature])
			
			# Determine if a song is instrumental/lyrical from Spotify features
			song_lyrical = False
			if audio_features['instrumentalness'] < 0.45 or audio_features['speechiness'] > 0.33:
				song_lyrical = True

			# Get Spotify link to tracks
			link = track['track']['external_urls']['spotify']

			now = time.time()
			if now - program_start > max_wait_time and counter == 0:
				st.markdown(f'Taking too long to find {option} {option_inst} songs for you in this playlist!')
				st.markdown('Try:\n* Choosing a different playlist \n* Toggling instrumental/vocal music')
				timed_out = True
				break
			elif now - program_start > max_wait_time and counter > 0:
				break

			# Use pretrained model based on whether instrumental or lyrical
			if option_inst == 'Lyrical/Vocal':
				if not song_lyrical:
					continue

				# Try to get lyrics for lyrical song; if can't, pass on the song
				try:
					song = genius.search_song(track_name, artist, get_full_info=False)
					# Ensure the song lyrics are for the correct song title,
					# and make sure the lyrics are in English
					if fuzz.token_set_ratio(song.title, track_name) > 80 \
                    and detect(re.sub(r'[^a-zA-z]', ' ', song.lyrics.lower())) == 'en':
						lyrics = re.sub(r'[^a-zA-Z]', ' ', song.lyrics.lower())
						sid = SentimentIntensityAnalyzer()
						sent = sid.polarity_scores(lyrics)
						sent_score = sent['compound']
						features.insert(0, sent_score)
					else:
						continue
				# Continue to next song if can't find lyrics with Genius API 
				except:
					continue
				
				scaled_features = pretrained_lyrical_scale.transform(np.array(features).reshape(1, -1))
				if pretrained_lyrical_rfc.predict(scaled_features)[0].lower() == option.lower():
				#if pretrained_lyrical_rfc.predict(np.array(features).reshape(1, -1))[0].lower() == option.lower():
					songs.append({ 'artist':artist, 'track_name':track_name, 'link':link, 'track_id':track_id })
				else:
					continue
			elif option_inst == 'Instrumental': 
				if song_lyrical:
					continue
				
				scaled_features = pretrained_inst_scale.transform(np.array(features).reshape(1, -1))
				if pretrained_inst_rfc.predict(scaled_features)[0].lower() == option.lower():
					songs.append({ 'artist':artist, 'track_name':track_name, 'link':link, 'track_id':track_id })
				else:
					continue			
	
			if counter == 4:
				break

			counter += 1

		if not timed_out:
			if len(songs):
				st.markdown(f'Here are some songs from the Spotify playlist that will make you feel ... {option}!  {emoji_dict[option]}')
				for song in songs:
					#st.markdown('[{}]({}) by {}'.format(song['track_name'], song['link'], song['artist']))
					embed_spotify_track(song['track_id'])
					#time.sleep(0.5)
			else:
				st.markdown(f'Could not find any {option} {option_inst} songs for you in this playlist!')
				st.markdown('Try:\n* Choosing a different playlist \n* Toggling instrumental/vocal music')

