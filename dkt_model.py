import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DKTDataset(Dataset):
    def __init__(self, sequences, max_seq_length):
        self.sequences = sequences
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_len = len(seq)

        # Pad sequence
        padded_seq = torch.zeros(self.max_seq_length, 2)
        padded_seq[:seq_len] = torch.FloatTensor(seq)

        # Create mask
        mask = torch.zeros(self.max_seq_length)
        mask[:seq_len] = 1

        return {
            'inputs': padded_seq[:-1],
            'targets': padded_seq[1:, 0],
            'skills': padded_seq[1:, 1],
            'mask': mask[1:]
        }


class DKTModel(nn.Module):
    def __init__(self, num_skills, hidden_size, num_layers=2):
        super(DKTModel, self).__init__()
        self.num_skills = num_skills
        self.hidden_size = hidden_size

        # Embedding layer for skills
        self.skill_embedding = nn.Embedding(num_skills + 1, hidden_size // 2)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Output layer
        self.output = nn.Linear(hidden_size, num_skills)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, skills, hidden=None):
        # Embed skills
        skill_emb = self.skill_embedding(skills.long())

        # Combine with input correctness
        inputs = inputs.unsqueeze(-1)
        combined = torch.cat([inputs, skill_emb], dim=-1)

        # LSTM processing
        lstm_out, hidden = self.lstm(combined, hidden)

        # Predict next skill mastery
        output = self.sigmoid(self.output(lstm_out))
        return output, hidden


def train_dkt_model(sequences, num_skills, epochs=10, batch_size=32):
    dataset = DKTDataset(sequences, max_seq_length=100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DKTModel(num_skills, hidden_size=128)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch['inputs']
            targets = batch['targets']
            skills = batch['skills']
            mask = batch['mask']

            optimizer.zero_grad()
            outputs, _ = model(inputs, skills)

            # Calculate masked loss
            loss = criterion(outputs.squeeze() * mask, targets * mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    return model