from moviepy.video.io.VideoFileClip import VideoFileClip
from pytube import YouTube, Playlist
import os
import time


def downloadVideo(playlist_videos, genre):
    count1 = 0
    count2 = 0
    for url in playlist_videos:
        yt = YouTube(url)
        yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
        new_name = genre + '_' + str(count1)
        os.rename(yt.streams.first().default_filename, new_name + '.mp4')

        # Load the mp4 file
        video = VideoFileClip(new_name + '.mp4')
        start = video.duration / 3
        end = start + 15

        # Extract audio from video
        video = video.subclip(start, end)
        video.audio.write_audiofile(new_name + '.mp3')

        video.close()
        os.remove(new_name + '.mp4')

        count1 += 1




def extract_urls(playlist):
    urls = []

    playlist_urls = Playlist(playlist)
    for url in playlist_urls:
        urls.append(url)

    return urls


playlist = 'https://www.youtube.com/watch?v=kXYiU_JCYtU&list=PLyORnIW1xT6wFALM5dZlkFhOULbToFok3'
pl_urls = extract_urls(playlist)
downloadVideo(pl_urls, 'rock')


