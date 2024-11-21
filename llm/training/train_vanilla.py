import tiktoken
import torch
from tqdm import tqdm

from llm.data.casual_dataloader import get_dataloader
from llm.data.utils import text_to_token_ids, token_ids_to_text, generate_text
from llm.models.gpt import GPTModelMy
from llm.training.train_utils import calc_loss_batch, calc_loss_loader


def generate_and_print_sample(model, tokenizer, device, start_context):
    context_size = model.pos_embeddings.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model,
            encoded,
            max_new_tokens=50,
            context_size=context_size,
            top_k=3,
            temperature=1.0,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    return decoded_text


def train_model_simple(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    num_epochs,
    start_context,
    tokenizer,
):
    train_losses = []
    val_losses = []
    track_token_seen = []
    tokens_seen = 0
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        bar = tqdm(train_dataloader)
        batch_losses = []
        for input_batch, target_batch in bar:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            bar.set_description(f"Train loss: {loss.item()}")
            batch_losses.append(loss)
        train_loss = torch.mean(torch.tensor(batch_losses)).item()
        train_losses.append(train_loss)
        model.eval()
        with torch.no_grad():
            val_loss = calc_loss_loader(val_dataloader, model, device)
        val_losses.append(val_loss)
        track_token_seen.append(tokens_seen)
        text = generate_and_print_sample(model, tokenizer, device, start_context)
        print(f"Epoch: {epoch}, train: {train_loss}, val: {val_loss}")
        print(text)
    return train_losses, val_losses, track_token_seen


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    file_path = "data/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_text = text_data[:split_idx]
    val_text = text_data[split_idx:]

    train_dataloader = get_dataloader(
        text=train_text,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_dataloader = get_dataloader(
        text=val_text,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = GPTModelMy(GPT_CONFIG_124M)
    device = torch.device("mps")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    tokenizer = tiktoken.get_encoding("gpt2")
    num_epochs = 10
    train_losses, val_losses, token_seen = train_model_simple(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        device,
        num_epochs,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )
