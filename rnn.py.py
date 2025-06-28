
# ------------------------- RNN -Script -------------------------
from main_module import main  

if __name__ == "__main__":
    main(model_type="rnn", epochs=25, batch_size=32, lr=2e-4)
class RNNModel_Modified(nn.Module):
    def __init__(self, rnn_type="rnn", embedding_dim=embedding_dim, hidden_dim=hiddendim):
        super(RNNModel_Modified, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) #vocab_size from simple_tokenizer
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN, "bilstm": nn.LSTM}[rnn_type]
        is_bidirectional = True if rnn_type in ["lstm", "bilstm"] else False # Note: RNN is NOT bidirectional by default here, as 'rnn' is not in ['lstm', 'bilstm']
        self.rnn = rnn_cls(embedding_dim, hidden_dim, batch_first=True, bidirectional=is_bidirectional)
        self.output_dim = hidden_dim * (2 if is_bidirectional else 1)
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
