import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from tokenizers import Tokenizer
from layers import Decoder, DecoderLayer, MultiheadAttention, FeedForward, Embedding


# 1. Архитектура модели
class GeneratorTransformer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, d_ff=1024, num_layers=4, vocab_size=32000,
                 pad_index=0, dropout=0.1, max_len=128, tokenizer=None, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.device = device
        self.pad_index = pad_index

        self.eos_id = tokenizer.token_to_id('</s>')
        self.bos_id = tokenizer.token_to_id('<s>')

        mha = MultiheadAttention(d_model, num_heads)
        enc_dec_mha = MultiheadAttention(d_model, num_heads)
        ffn = FeedForward(d_model, d_ff)

        self.decoder = Decoder(DecoderLayer(mha, enc_dec_mha, ffn, dropout), num_layers)
        self.normalize = nn.LayerNorm(d_model)
        self.embedding = Embedding(d_model, vocab_size, pad_index)
        self.vocab_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        x = self.embedding(x)
        x = self.decoder(x, x, None, mask)
        x = self.normalize(x)
        return self.vocab_projection(x)

    def generate(self, prompt, context_len=None, temperature=1.0, max_out_tokens=200):
        '''Авторегрессивная генерация со сдвигом контекста'''
        if context_len is None:
            context_len = self.max_len

        self.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt).ids
            input_ids = [self.bos_id] + input_ids
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

            generated = input_ids.clone()

            for _ in range(max_out_tokens):
                curr_input = generated[:, -context_len:]

                outputs = self(curr_input)
                next_token_logits = outputs[0, -1, :] / temperature

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                next_token = next_token.unsqueeze(0)

                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == self.eos_id:
                    break

        # Декодируем, обрезая специальные токены для красоты вывода
        tokens = generated[0].tolist()
        # Убираем BOS в начале
        if tokens and tokens[0] == self.bos_id:
            tokens = tokens[1:]
        # Убираем EOS в конце
        if tokens and tokens[-1] == self.eos_id:
            tokens = tokens[:-1]
        # Убираем PAD
        tokens = [t for t in tokens if t != self.pad_index]

        return self.tokenizer.decode(tokens)

    def generate_beam(self, prompt, beam_width=3, max_out_tokens=50, temperature=1.0):
        '''Генерация с использованием Beam Search'''
        self.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt).ids
            input_ids = [self.bos_id] + input_ids
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

            beams = [(input_ids, 0.0)]
            completed_beams = []

            for _ in range(max_out_tokens):
                all_candidates = []
                for seq, score in beams:
                    if seq[0, -1].item() == self.eos_id:
                        completed_beams.append((seq, score))
                        continue

                    curr_input = seq[:, -self.max_len:]
                    outputs = self(curr_input)
                    logits = outputs[0, -1, :] / temperature
                    log_probs = torch.log_softmax(logits, dim=-1)

                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)

                    for i in range(beam_width):
                        new_seq = torch.cat([seq, topk_ids[i].unsqueeze(0).unsqueeze(0)], dim=1)
                        new_score = score + topk_log_probs[i].item()
                        all_candidates.append((new_seq, new_score))

                if not all_candidates:
                    break

                all_candidates.sort(key=lambda x: x[1], reverse=True)
                beams = all_candidates[:beam_width]

            completed_beams.extend(beams)
            completed_beams.sort(key=lambda x: x[1], reverse=True)
            best_seq = completed_beams[0][0]

        tokens = best_seq[0].tolist()
        if tokens and tokens[0] == self.bos_id:
            tokens = tokens[1:]
        if tokens and tokens[-1] == self.eos_id:
            tokens = tokens[:-1]
        tokens = [t for t in tokens if t != self.pad_index]

        return self.tokenizer.decode(tokens)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, tokenizer, device='cuda'):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        model = cls(
            d_model=config['d_model'], num_heads=config['nhead'], d_ff=config['d_ff'],
            num_layers=config['num_layers'], vocab_size=tokenizer.get_vocab_size(),
            pad_index=tokenizer.token_to_id('<pad>'), dropout=config.get('dropout', 0.1),
            max_len=config['max_length'], tokenizer=tokenizer, device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)


# 2. Датасет

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.token_to_id('<pad>')
        self.bos_id = tokenizer.token_to_id('<s>')
        self.eos_id = tokenizer.token_to_id('</s>')

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        self.chunks = []
        for p in paragraphs:
            tokens = tokenizer.encode(p).ids
            if len(tokens) > max_length - 2:
                tokens = tokens[:max_length - 2]

            tokens = [self.bos_id] + tokens + [self.eos_id]

            if len(tokens) < max_length:
                tokens = tokens + [self.pad_id] * (max_length - len(tokens))

            self.chunks.append(tokens)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx], dtype=torch.long)


# 3. Обучение
class GeneratorTrainer:
    def __init__(self, model, train_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.pad_id = model.pad_index

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-4))
        self.save_dir = config.get('save_dir', 'checkpoints_gen')
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        scaler = GradScaler()

        for i, batch in enumerate(progress_bar):
            batch = batch.to(self.device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            self.optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = self.model(inputs)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{(total_loss / (i + 1)):.4f}'})

        return total_loss / max(1, i)

    def train(self):
        num_epochs = self.config.get('num_epochs', 10)
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            loss = self.train_epoch(epoch)
            print(f'Train - Loss: {loss:.4f}')
            self.save_checkpoint(f'epoch_{epoch + 1}.pt')

    def save_checkpoint(self, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, path)
        print(f'Checkpoint saved: {path}')



# 4. Чат-интерфейс

def chat(model_path="checkpoints_gen/epoch_2.pt", tokenizer_path="mistral_tokenizer.json"):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.add_special_tokens(['<pad>', '<s>', '</s>'])
    model = GeneratorTransformer.load_from_checkpoint(model_path, tokenizer)
    model.eval()

    print("\n=== Чат с моделью (введите 'quit' для выхода) ===")
    while True:
        user_input = input("\nВы: ")
        if user_input.lower() == 'quit':
            break

        if not user_input.strip():
            continue

        try:
            response = model.generate(user_input, temperature=0.8, max_out_tokens=100)
            print(f"Бот (Greedy): {response}")

            response_beam = model.generate_beam(user_input, beam_width=3, max_out_tokens=50)
            print(f"Бот (Beam):   {response_beam}")
        except Exception as e:
            print(f"️ Ошибка генерации: {e}")
            print("Попробуйте другой промпт.")


# Main
if __name__ == "__main__":
    text_file = "data.txt"
    tokenizer_path = "mistral_tokenizer.json"

    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Файл {text_file} не найден.")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Файл {tokenizer_path} не найден.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    config = {
        'batch_size': 16,
        'max_length': 128,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'd_ff': 1024,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'num_epochs': 2,
        'save_dir': 'checkpoints_gen',
    }

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.add_special_tokens(['<pad>', '<s>', '</s>'])

    dataset = TextDataset(text_file, tokenizer, max_length=config['max_length'])
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = GeneratorTransformer(
        d_model=config['d_model'], num_heads=config['nhead'], d_ff=config['d_ff'],
        num_layers=config['num_layers'], vocab_size=tokenizer.get_vocab_size(),
        pad_index=tokenizer.token_to_id('<pad>'), dropout=config['dropout'],
        max_len=config['max_length'], tokenizer=tokenizer, device=device
    )
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    trainer = GeneratorTrainer(model, train_loader, config, device)
    trainer.train()

    chat(model_path="checkpoints_gen/epoch_2.pt", tokenizer_path=tokenizer_path)