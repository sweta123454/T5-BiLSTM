
# ------------------------- GRU - Script -------------------------
from main_module import main 

if __name__ == "__main__":
    main(model_type="gru", epochs=25, batch_size=32, lr=2e-4)
if model_type in ["rnn", "bilstm", "gru"]:
        tokenizer = SimpleTokenizer(df['original'].tolist() + df['augmented'].tolist())
        is_transformer = False
        model = RNNModel(rnn_type=model_type, vocab_size=tokenizer.vocab_size()).to(device)

tokenizer = SimpleTokenizer(df['original'].tolist() + df['augmented'].tolist())
# For bilstm: model = RNNModel(rnn_type="bilstm", vocab_size=tokenizer.vocab_size()).to(device)
# For gru: model = RNNModel(rnn_type="gru", vocab_size=tokenizer.vocab_size()).to(device)
# For rnn: model = RNNModel(rnn_type="rnn", vocab_size=tokenizer.vocab_size()).to(device)
is_transformer = False # Make sure this is set correctly

