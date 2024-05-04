import torch
import torchaudio
from neuralNetwork import CNNNetwork
from audioDataset import CustomSongDataset
from training import AUDIO_DIRECTORY, ANNOTATION_FILE, SAMPLE_RATE, NUM_SAMPLES

# tells us what our class labels should actually be for the corresponding indices
genres = [
    "country",
    "pop",
    "rock",
    "jazz",
    "metal"
]


# actual function that predicts with our test data
def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> i.e. [ [0.1, 0.02, ..., 0.25] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load up the model
    cnn = CNNNetwork()
    state_dict = torch.load("cnn_audio.pth")
    cnn.load_state_dict(state_dict)

    # load the CustomSongDataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )


    songs_dataset = CustomSongDataset(ANNOTATION_FILE,
                                      AUDIO_DIRECTORY,
                                      mel_spectrogram,
                                      SAMPLE_RATE,
                                      NUM_SAMPLES)

    # get a testing sample from the validation dataset for inference, add an extra dimension because batchsize = 1
    input, target = songs_dataset[303][0], songs_dataset[303][1]
    input.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(cnn, input, target, genres)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
