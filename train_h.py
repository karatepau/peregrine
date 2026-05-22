import sys
from model import heuristic
import pycuber as pc
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
heuristic.to(device)

def train (epochs):
    with open('dataset.pkl', 'rb') as f:
        cubes, solutions = pickle.load(f)
    size = len(solutions)
    x = torch.tensor(cubes, dtype=torch.float32)
    y = torch.tensor(solutions, dtype=torch.long)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(heuristic.parameters(), lr=0.0001)
    with torch.no_grad():
        initial_loss = loss_function(heuristic(x.to(device)), y.to(device)).item()
    print(f"Pérdida inicial: {initial_loss:.4f}")

    for epoch in range(epochs):
        accumulated_error = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predicciones = heuristic(batch_X)
            loss = loss_function(predicciones, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            accumulated_error += loss.item()
        avg_error = accumulated_error / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} -> Loss: {avg_error:.4f}")
    return avg_error

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Using: {device}")
    try:
        heuristic.load_state_dict(torch.load('heuristic.pth', map_location=device))
        print("Pesos cargados OK")
    except FileNotFoundError:
        pass
    print(train(epochs))
    torch.save(heuristic.state_dict(), 'heuristic.pth')
    print("Completed")
