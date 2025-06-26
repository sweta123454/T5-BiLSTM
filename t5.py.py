
# ------------------------- T5 - Script -------------------------
from main_module import main 

if __name__ == "__main__":
    main(model_type="t5", epochs=25, batch_size=32, lr=2e-4)
class T5EncoderWrapper(nn.Module):
    def __init__(self, rnn_type=None):
        super(T5EncoderWrapper, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained("t5-base")
        for name, param in self.encoder.named_parameters():
            if "block.0" in name or "block.1" in name or "block.2" in name:
                param.requires_grad = False
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.rnn_type = rnn_type
        if rnn_type:
            rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN, "bilstm": nn.LSTM}[rnn_type]
            is_bidirectional = True if rnn_type in ["lstm", "bilstm"] else False
            self.rnn = rnn_cls(projection_dim, hidden_dim, batch_first=True, bidirectional=is_bidirectional)
            self.output_dim = hidden_dim * (2 if is_bidirectional else 1)
        else:
            self.output_dim = projection_dim
        self.attn_pool = AttentionPooling(self.output_dim)
        self.mlp_head = MLPHead(self.output_dim * 3)

    def forward(self, input_ids1, input_ids2, attention_mask1, attention_mask2):
        emb1 = self.projection(self.encoder(input_ids1.to(device), attention_mask=attention_mask1.to(device)).last_hidden_state)
        emb2 = self.projection(self.encoder(input_ids2.to(device), attention_mask=attention_mask2.to(device)).last_hidden_state)
        if self.rnn_type:
            x1, _ = self.rnn(emb1)
            x2, _ = self.rnn(emb2)
        else:
            x1, x2 = emb1, emb2
        h1 = self.attn_pool(x1)
        h2 = self.attn_pool(x2)
        fusion = torch.cat([h1, h2, torch.abs(h1 - h2)], dim=1)
        return self.mlp_head(fusion)