import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Mess3Process:
    def __init__(self, alpha: float, x: float):
        self.alpha = alpha
        self.x = x
        self.beta = (1 - alpha) / 2
        self.y = 1 - 2 * x
        
        # Token-labeled transition matrices T^(0), T^(1), T^(2)
        self.T0 = np.array([
            [self.alpha * self.y, self.beta * self.x, self.beta * self.x],
            [self.alpha * self.x, self.beta * self.y, self.beta * self.x],
            [self.alpha * self.x, self.beta * self.x, self.beta * self.y]
        ])
        
        self.T1 = np.array([
            [self.beta * self.y, self.alpha * self.x, self.beta * self.x],
            [self.beta * self.x, self.alpha * self.y, self.beta * self.x],
            [self.beta * self.x, self.alpha * self.x, self.beta * self.y]
        ])
        
        self.T2 = np.array([
            [self.beta * self.y, self.beta * self.x, self.alpha * self.x],
            [self.beta * self.x, self.beta * self.y, self.alpha * self.x],
            [self.beta * self.x, self.beta * self.x, self.alpha * self.y]
        ])
        
        self.T = self.T0 + self.T1 + self.T2
        
        # The stationary distribution is uniform: [1/3, 1/3, 1/3]
        self.state_dist = np.array([1/3, 1/3, 1/3])

    def generate_sequence(self, length: int) -> list[int]:
        """Generates a sequence of tokens from this specific Mess3 process."""
        sequence = []
        # Sample initial hidden state from the stationary distribution
        current_state = np.random.choice([0, 1, 2], p=self.state_dist)
        
        for _ in range(length):
            # Calculate probs of emitting 0, 1, or 2 given current state
            # i.e. sum over the possible next states for each token matrix
            p0 = np.sum(self.T0[current_state, :])
            p1 = np.sum(self.T1[current_state, :])
            p2 = np.sum(self.T2[current_state, :])
            
            # Normalize to ensure valid probability distribution
            probs = np.array([p0, p1, p2])
            probs = probs / np.sum(probs)
            
            # Sample the token
            token = np.random.choice([0, 1, 2], p=probs)
            sequence.append(token)
            
            # Update the hidden state based on emitted token
            if token == 0:
                trans_matrix = self.T0
            elif token == 1:
                trans_matrix = self.T1
            else:
                trans_matrix = self.T2
                
            next_state_probs = trans_matrix[current_state, :]
            next_state_probs = next_state_probs / np.sum(next_state_probs)
            current_state = np.random.choice([0, 1, 2], p=next_state_probs)
            
        return sequence

class NonErgodicMess3Dataset(Dataset):
    def __init__(self, num_samples: int, seq_length: int, processes: list[Mess3Process]):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.processes = processes
        self.num_processes = len(processes)
        
        self.data = []
        self.process_labels = []
        
        for _ in range(num_samples):
            process_idx = np.random.randint(0, self.num_processes)
            process = self.processes[process_idx]
            
            seq = process.generate_sequence(seq_length)
            self.data.append(seq)
            self.process_labels.append(process_idx)
            
        self.data = torch.tensor(self.data, dtype=torch.long)
        self.process_labels = torch.tensor(self.process_labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return input sequence and target sequence (shifted by 1 for next-token prediction)
        seq = self.data[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y, self.process_labels[idx]

if __name__ == "__main__":
    # Define K=3 different Mess3 processes (using parameters similar to the paper's variants)
    p1 = Mess3Process(alpha=0.60, x=0.15)
    p2 = Mess3Process(alpha=0.79, x=0.11)
    p3 = Mess3Process(alpha=0.60, x=0.50)
    
    dataset = NonErgodicMess3Dataset(num_samples=1000, seq_length=17, processes=[p1, p2, p3])
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    x, y, labels = next(iter(loader))
    print(f"Input batch shape: {x.shape}")
    print(f"Target batch shape: {y.shape}")
    print(f"Labels shape: {labels.shape}")