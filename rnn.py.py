
# ------------------------- RNN -Script -------------------------
from main_module import main  

if __name__ == "__main__":
    main(model_type="rnn", epochs=25, batch_size=32, lr=2e-4)
class RNNModel(nn.Module):
    def __init__(self, rnn_type, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, bidirectional=True):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN, "bilstm": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        self.attn_pool = AttentionPooling(self.output_dim)
        self.mlp_head = MLPHead(self.output_dim * 3)

    def forward(self, input_ids1, input_ids2, attention_mask1=None, attention_mask2=None):
        emb1 = self.embedding(input_ids1.to(device))
        emb2 = self.embedding(input_ids2.to(device))

        x1, _ = self.rnn(emb1)
        x2, _ = self.rnn(emb2)

        h1 = self.attn_pool(x1)
        h2 = self.attn_pool(x2)
        fusion = torch.cat([h1, h2, torch.abs(h1 - h2)], dim=1)
        return self.mlp_head(fusion)
