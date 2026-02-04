import torch
import torch.nn as nn

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(TextGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# Load the model
model_path = "text generator.pth"

# Load state_dict
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Infer sizes from state_dict
vocab_size = state_dict['fc.weight'].shape[0]
embed_size = state_dict['embedding.weight'].shape[1]
hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4  # LSTM has 4 gates
num_layers = 1

print(f"Vocab size: {vocab_size}, Embed size: {embed_size}, Hidden size: {hidden_size}")

model = TextGenerator(vocab_size, embed_size, hidden_size, num_layers)
model.load_state_dict(state_dict)
model.eval()

# For character-level generation, assuming ASCII or custom vocab
# If vocab_size > 128, it might be a different encoding
if vocab_size <= 128:
    char_to_int = {chr(i): i for i in range(vocab_size)}
    int_to_char = {i: chr(i) for i in range(vocab_size)}
else:
    # Assume indices 0 to vocab_size-1, map to printable chars or something
    char_to_int = {str(i): i for i in range(vocab_size)}
    int_to_char = {i: str(i) for i in range(vocab_size)}

def generate_text(start_text, length=100, temperature=1.0):
    model.eval()
    with torch.no_grad():
        # Convert start text to indices, using 0 for unknown chars
        input_seq = torch.tensor([[char_to_int.get(c, 0) for c in start_text]], dtype=torch.long)
        
        hidden = None
        generated = start_text
        
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            output = output[:, -1, :] / temperature
            probs = torch.softmax(output, dim=-1)
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = int_to_char.get(next_char_idx, '?')
            generated += next_char
            input_seq = torch.tensor([[next_char_idx]], dtype=torch.long)
        
        return generated

if __name__ == "__main__":
    start_text = "Hello"
    generated = generate_text(start_text, length=50)
    print(f"Generated text: {generated}")
