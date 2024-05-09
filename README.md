# Project File Descriptions

## annotations_generator.py

Takes a folder path and processes all files within that folder. It extracts information from each file name based on specific criteria (like prefixes such as "country_", "pop_", etc.), assigns numerical IDs accordingly, and then creates a CSV file with the processed data. Finally, it prints a success message once the file writing process is completed.

## dataset.py

Defines a custom PyTorch dataset GenreDataset for working with audio data. It reads annotations from a CSV file, loads audio samples using torchaudio, applies transformations like resampling and mixing down, and returns the transformed audio signal along with its corresponding label. When executed as the main script, it creates an instance of GenreDataset, initializes audio transformations (Mel spectrogram in this case), and prints information about the dataset such as the number of samples and the first sample along with its label.

## neural_net.py

Defines a convolutional neural network (CNN) architecture using PyTorch's nn.Module. The network comprises four convolutional blocks followed by a flatten layer and a linear layer for classification. The forward method performs the forward pass through the network, and when executed as the main script, it creates an instance of the CNNNetwork and prints a summary of the model, indicating the layer configurations and output shapes.

## playlist_scraper.py

Uses the MoviePy library and the PyTube library to download videos from a YouTube playlist, extract audio from them, and save them as MP3 files. The downloadPlaylistAsAudio function takes a list of video URLs from a playlist, downloads each video, extracts a 15-second audio clip from the middle third of the video, and saves it as an MP3 file with a naming convention based on the genre. The extract_urls function extracts video URLs from a given playlist URL using the PyTube Playlist class. Finally, the run_scraper function orchestrates the downloading and audio extraction process for a given playlist URL and genre.

## plotting.py

Uses the librosa library for loading audio files, transforms the audio data into the desired format (e.g., mel spectrogram), and provides methods to plot the waveform and mel spectrogram of audio samples. The script also includes helper functions for handling audio signal cutting and padding. When executed as the main script, it creates an instance of CustomSongDataset, loads audio samples from the specified directory, and demonstrates how to plot waveform and mel spectrogram visualizations for a selected audio sample.

## training.py

Trains a neural network model for audio classification using PyTorch. It defines hyperparameters such as batch size, epochs, and learning rate, loads a custom dataset GenreDataset for audio data, creates a data loader for training, instantiates a convolutional neural network (CNN) model CNNNetwork, sets up a loss function (CrossEntropyLoss), an optimizer (Adam), and then trains the model for the specified number of epochs. Finally, it saves the trained model to a file.
