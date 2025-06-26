
# ------------------------- DISTILBERT - Script -------------------------
from main_module import main  

if __name__ == "__main__":
    main(model_type="distilbert", epochs=25, batch_size=32, lr=2e-4)
class DistilBertWrapper(nn.Module):
    def __init__(self):
        super(DistilBertWrapper, self).__init__()
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        for name, param in self.encoder.named_parameters():
            if name.startswith("distilbert.transformer.layer.0") or name.startswith("distilbert.transformer.layer.1") or name.startswith("distilbert.transformer.layer.2"):
                param.requires_grad = False
        self.projection = nn.Linear(768, projection_dim)
        self.attn_pool = AttentionPooling(projection_dim)
        self.mlp_head = MLPHead(projection_dim * 3)

    def forward(self, input_ids1, input_ids2, attention_mask1, attention_mask2):
        emb1 = self.projection(self.encoder(input_ids1.to(device), attention_mask=attention_mask1.to(device)).last_hidden_state)
        emb2 = self.projection(self.encoder(input_ids2.to(device), attention_mask=attention_mask2.to(device)).last_hidden_state)
        h1 = self.attn_pool(emb1)
        h2 = self.attn_pool(emb2)
        fusion = torch.cat([h1, h2, torch.abs(h1 - h2)], dim=1)
        return self.mlp_head(fusion)