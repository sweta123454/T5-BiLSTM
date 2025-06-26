from main_module import main 
class main(nn.Module):
    def main(model_type="t5+bilstm", epochs=25, batch_size=32, lr=2e-4):
        print(f"\n===== Starting training for model: {model_type} =====")
        df = pd.read_csv("/home/swetas/hom1/model/squad_augmented_18k.csv")

        if model_type in ["rnn", "bilstm", "gru"]:
            tokenizer = SimpleTokenizer(df['original'].tolist() + df['augmented'].tolist())
            is_transformer = False
            model = RNNModel(rnn_type=model_type, vocab_size=tokenizer.vocab_size()).to(device)
        elif model_type == "distilbert":
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            is_transformer = True
            model = DistilBertWrapper().to(device)
        else:
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
            rnn_type = model_type.split('+')[1] if '+' in model_type else None
            model = T5EncoderWrapper(rnn_type=rnn_type).to(device)
            is_transformer = True

        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        train_ds = SimilarityDataset(train_df, tokenizer, is_transformer)
        val_ds = SimilarityDataset(val_df, tokenizer, is_transformer)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=patience)

        train_losses, val_losses = [], []
        train_rmse_list, val_rmse_list = [], []
        train_r2_list, val_r2_list = [], []
        train_pearson_list, val_pearson_list = [], []
        train_spearman_list, val_spearman_list = [], []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs} for {model_type}")
            train_loss, train_preds, train_trues = train_model(model, train_loader, optimizer, scheduler, criterion)
            val_preds, val_trues, val_loss = evaluate_model(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_metrics = calculate_metrics(train_preds, train_trues)
            val_metrics = calculate_metrics(val_preds, val_trues)

            train_rmse_list.append(train_metrics['RMSE'])
            train_r2_list.append(train_metrics['R2'])
            train_pearson_list.append(train_metrics['Pearson'])
            train_spearman_list.append(train_metrics['Spearman'])

            val_rmse_list.append(val_metrics['RMSE'])
            val_r2_list.append(val_metrics['R2'])
            val_pearson_list.append(val_metrics['Pearson'])
            val_spearman_list.append(val_metrics['Spearman'])

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print("Train:")
            print_eval_metrics(train_preds, train_trues)
            print("Validation:")
            print_eval_metrics(val_preds, val_trues)

            if early_stopping(val_loss):
                print("Early stopping triggered.")
                break

        title = model_type
        plot_loss_curves(train_losses, val_losses, title)
        plot_evaluation_metrics(train_rmse_list, train_r2_list, train_pearson_list, train_spearman_list, title, "Train")
        plot_evaluation_metrics(val_rmse_list, val_r2_list, val_pearson_list, val_spearman_list, title, "Validation")

if __name__ == "__main__":
    for model_type in [
        "t5", "t5+bilstm", "t5+gru", "t5+rnn",
        "bilstm", "gru", "rnn", "distilbert"]:
        main(model_type=model_type, epochs=25, batch_size=32, lr=2e-4)