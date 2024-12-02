import spotipy
import streamlit
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

from src.models.service_result import ServiceResult


class SpotifyClient:
    def __init__(self, client_id, client_secret):
        """
        Initializes the SpotifyClient with client credentials.
        """

        self.__client_id = client_id
        self.__client_secret = client_secret
        scope = "user-library-read playlist-modify-private playlist-modify-public"
        auth_manager = SpotifyOAuth(client_id=client_id,
                                    client_secret=client_secret,
                                    redirect_uri="http://localhost:8051",
                                    scope=scope)
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    def search_track(self, query):
        """
        Search for a track based on a query string.`
        """
        results = self.sp.search(q=query, type='track', limit=1)
        tracks = results['tracks']['items']
        if tracks:
            first_track = tracks[0]
            track_info = {
                'name': first_track['name'],
                'artists': [artist['name'] for artist in first_track['artists']],
                'external_url': first_track['external_urls']['spotify']
            }
            return track_info
        else:
            return "No tracks found."

    @streamlit.cache_data(hash_funcs={"src.clients.spotify_client.SpotifyClient": lambda x: hash(x.__client_id + x.__client_secret),
                                      str: lambda uri: hash(uri)})
    def get_spotify_tracks_by_playlist_uri(self, playlist_uri) -> ServiceResult:
        try:
            playlist_tracks = self.sp.playlist_items(playlist_uri)
            return ServiceResult(True, playlist_tracks)

        except Exception as e:
            return ServiceResult(False, None, str(e))

    @streamlit.cache_data(hash_funcs={"src.clients.spotify_client.SpotifyClient": lambda x: hash(x.__client_id + x.__client_secret),
                                      str: lambda uri: hash(uri)})
    def get_playlist_info_by_playlist_uri(self, playlist_uri) -> ServiceResult:
        try:
            playlist_info = self.sp.playlist(playlist_uri)
            return ServiceResult(True, playlist_info)

        except Exception as e:
            return ServiceResult(False, None, str(e))

    @streamlit.cache_data(
        hash_funcs={"src.clients.spotify_client.SpotifyClient": lambda x: hash(x.__client_id + x.__client_secret),
                    list[str]: lambda track_ids: hash(tuple(track_ids))})
    def create_playlist(self, track_ids, playlist_name="My New Playlist"):
        try:
            print("I AM CALLED")
            user_id = self.sp.current_user()["id"]
            playlist = self.sp.user_playlist_create(user=user_id, name=playlist_name, public=True)
            playlist_id = playlist["id"]
            self.sp.playlist_add_items(playlist_id, track_ids)
            return ServiceResult(True, playlist)
        except Exception as e:
            return ServiceResult(False, None, str(e))

    @staticmethod
    def get_playlist_uri_by_raw_link(playlist_link):
        return playlist_link[34:56]
