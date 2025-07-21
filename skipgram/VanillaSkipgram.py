from torch import nn

class VanillaSkipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # Initialize the embedding layer with vocab size and embedding dimension
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        # Initialize the linear layer to project embeddings to vocab size
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        output = self.linear(embeddings)
        return output
    