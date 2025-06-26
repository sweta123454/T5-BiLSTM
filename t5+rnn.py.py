
# ------------------------- T5+RNN - Script -------------------------
from main_module import main  

if __name__ == "__main__":
    main(model_type="t5+rnn", epochs=25, batch_size=32, lr=2e-4)
tokenizer = T5Tokenizer.from_pretrained("t5-base")
rnn_type = model_type.split('+/_')[1] if '+/_' in model_type else None
model = T5EncoderWrapper(rnn_type=rnn_type).to(device)
# For t5_gru: model = T5EncoderWrapper(rnn_type="gru").to(device)
# For t5_rnn: model = T5EncoderWrapper(rnn_type="rnn").to(device)
# For t5_bilstm: model = T5EncoderWrapper(rnn_type="bilstm").to(device)
is_transformer = True

