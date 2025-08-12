# src/spotify_client.py
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load environment variables from .env
load_dotenv()

def get_spotify_client(client_id=None, client_secret=None):
    """
    Returns a Spotipy Spotify client.
    Reads credentials from arguments or from environment variables.
    """
    client_id = client_id or os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = client_secret or os.getenv("SPOTIPY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "Spotify credentials not found. "
            "Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in your .env file."
        )

    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp
