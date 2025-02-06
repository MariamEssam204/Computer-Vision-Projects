import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        #This line freezes all the parameters of the ResNet model,
        #which means that during training, these parameters will not be updated (i.e., the gradients won't be computed for them).
        #This is useful when you’re using ResNet as a feature extractor and don’t need to train it further.

        for param in resnet.parameters():
            param.requires_grad_(False)
            
        #This line removes the final fully connected layer (classification layer) of ResNet-50.
        # We don’t need this layer because we’re only interested in the feature map that the model produces before this layer
        modules = list(resnet.children())[:-1]
        
        #creates CNN without final fully connected layer 
        self.resnet = nn.Sequential(*modules)
        #maps the high-dimensional feature vector into a fixed-size embedding
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        #self.embed reduces it to embed_size (e.g., 256 or 512) to match the LSTM input size.
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
       
        #it's essential if your class inherits from another class and you need to initialize the parent class.
        #This ensures that the child class inherits all necessary attributes and behaviors from the parent class.
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
         # Word embedding layer for captions
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=0.5, batch_first=True)
        #Maps the hidden states to vocabulary size.
        self.fcl = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        """
        - features: CNN output (batch_size, embed_size)
        - captions: Word embeddings (batch_size, seq_length, embed_size)
        "The code correctly prepares the input for the LSTM by concatenating the image features with the word embeddings"
        """
        # Convert captions to embeddings
        embeddings = self.embedding(captions)  # (batch_size, seq_length, embed_size)

        # Expand the image features to match LSTM input dimensions
        features = features.unsqueeze(1)  # (batch_size, 1, embed_size)

        # Concatenate image features with caption embeddings
        lstm_input = torch.cat((features, embeddings), dim=2)  # (batch_size, seq_length+1, embed_size)

        # Pass through LSTM
        out, _ = self.lstm(lstm_input)

        # Map LSTM output to vocabulary size
        output = self.fc(out)
        return output
    
   
    def sample(self, inputs, states=None, max_len=20):
        """
        Generate a caption (sequence of words) for the given image (input tensor).
        accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len).
   
        Args:
            inputs: The pre-processed image tensor (batch_size, embed_size).
            states: The LSTM hidden states (optional), used to maintain the state across time steps.
            max_len: The maximum length of the generated caption (default is 20).
        
        Returns:
            predicted_caption: A list of word indices representing the predicted sentence (list of tensor IDs).
        """
        # Initialize a list to store the predicted words
        predicted_caption = []

        # Process the image through the CNN to extract features
        features = self.resnet(inputs)  # (batch_size, embed_size)
        features = features.view(features.size(0), -1)  # Flatten the feature map

        # Map the CNN features to the embedding space
        features = self.embed(features)  # (batch_size, embed_size)

        # Initialize the input to the LSTM with the start word embedding
        word = torch.tensor([self.vocab(self.start_word)]).unsqueeze(0)  # (1, 1) tensor for <start> token
        
        # Expand the image features for concatenation with the word embeddings
        features = features.unsqueeze(1)  # (batch_size, 1, embed_size)

        # Generate the caption word by word
        for _ in range(max_len):
            # Concatenate the image features with the current word embedding
            lstm_input = torch.cat((features, self.embedding(word)), dim=2)  # (batch_size, 1, embed_size + embed_size)

            # Forward pass through LSTM
            out, states = self.lstm(lstm_input, states)  # (batch_size, 1, hidden_size)

            # Predict the next word from the LSTM output
            output = self.fcl(out)  # (batch_size, 1, vocab_size)

            # Get the index of the predicted word (the word with the highest probability)
            _, predicted = output.max(2)  # (batch_size, 1), get index of highest probability word

            # Add the predicted word index to the caption list
            predicted_caption.append(predicted.item())

            # Set the predicted word as the input for the next step
            word = predicted

            # If the predicted word is the <end> token, stop generating
            if word.item() == self.vocab(self.end_word):
                break

        return predicted_caption
