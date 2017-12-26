# RNN notes
## Hyperparameters
hidden_size = 100
sequence_length = 25 # Output size
learning_rate = 0.1 # How quickly does the network abandon its current belief. If the network is being trained on cat
images and an image of a dog is shown, for a small LR, it will have a tendency to ignore the dog image as an anomaly.

# Bias: Allows us to move the line up and down to better fit the data. Else the line will always pass through the origin
and might give a poor result. It's kinda anchor value.

# 38.40: Something about LSTM networks \