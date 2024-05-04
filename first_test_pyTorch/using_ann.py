import torch
from first_ann import FeedForwardNet, download_mnist_datasets

# tells us what our class labels should actually be for the corresponding indices
# in the case of MNIST data, these indices are the same
class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
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
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    # load the MNIST validation dataset
    _, validation_data = download_mnist_datasets()

    # get a testing sample from the validation dataset for inference
    input, target = validation_data[44][0], validation_data[44][1]

    # make an inference
    predicted, expected = predict(feed_forward_net, input, target, class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
