import time

from moviepy.video.io.VideoFileClip import VideoFileClip
from pytube import YouTube, Playlist
import os


def downloadVideo(playlist_videos):
    count1 = 0
    count2 = 0
    for url in playlist_videos:
        yt = YouTube(url)
        yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
        new_name = 'file_' + str(count1)
        os.rename(yt.streams.first().default_filename, new_name + '.mp4')

        # Load the mp4 file
        video = VideoFileClip(new_name + '.mp4')
        start = video.duration / 3
        end = start + 15

        # Extract audio from video
        video = video.subclip(start, end)
        video.audio.write_audiofile(new_name + '.mp3')

        count1 += 1

    time.sleep(5)
    for i in playlist_videos:
        os.remove('file_' + str(count2) + '.mp4')
        count2 += 1


def extract_urls(playlist):
    urls = []

    playlist_urls = Playlist(playlist)
    for url in playlist_urls:
        urls.append(url)

    return urls


playlist = 'https://www.youtube.com/watch?v=lvhHZTlMKRU&list=PLXBNvqGzerDu1v7RsikFmN_XJCva5AAD-'
pl_urls = extract_urls(playlist)
downloadVideo(pl_urls)
