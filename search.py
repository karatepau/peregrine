import sys
from model import heuristic
import pycuber as pc
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

cache = {}

def normalizer(cube):
    normalized_cube = []
    FACES = ["U", "D", "F", "B", "R", "L"]
    COLOR_TO_ONE_HOT = {
        "white":  [1, 0, 0, 0, 0, 0],
        "yellow": [0, 1, 0, 0, 0, 0],
        "red":    [0, 0, 1, 0, 0, 0],
        "orange": [0, 0, 0, 1, 0, 0],
        "green":  [0, 0, 0, 0, 1, 0],
        "blue":   [0, 0, 0, 0, 0, 1]
    }
    for j, face in enumerate(FACES):
        for row in cube.get_face(face):
            for sticker in row:
                normalized_cube.extend(COLOR_TO_ONE_HOT[sticker.colour])
    return normalized_cube

def h (cb):
    key = str(cb)
    if key in cache:
        return cache[key]
    cube = normalizer(cb)
    with torch.no_grad():
        tensor = torch.tensor(cube, dtype=torch.float32)
        logits = heuristic(tensor.unsqueeze(0))
        result = logits.argmax().item()
        cache[key] = result
        return result

def dfs (cube, g, limit, path):
    f = g + h(cube)
    if f > limit:
        return f
    if cube == pc.Cube():
        return path
    sons = {
        "U": 0, "D": 0, "R": 0,
        "L": 0, "F": 0, "B": 0,
        "U'": 0, "D'": 0, "R'": 0,
        "L'": 0, "F'": 0, "B'": 0,
        "U2": 0, "D2": 0, "R2": 0,
        "L2": 0, "F2": 0, "B2": 0
    }
    excluded = {
        "U": ["U", "U'", "U2"], "D": ["D", "D'", "D2"], "R": ["R", "R'", "R2"],
        "L": ["L", "L'", "L2"], "F": ["F", "F'", "F2"], "B": ["B", "B'", "B2"],
        "U'": ["U", "U'", "U2"], "D'": ["D", "D'", "D2"], "R'": ["R", "R'", "R2"],
        "L'": ["L", "L'", "L2"], "F'": ["F", "F'", "F2"], "B'": ["B", "B'", "B2"],
        "U2": ["U", "U'", "U2"], "D2": ["D", "D'", "D2"], "R2": ["R", "R'", "R2"],
        "L2": ["L", "L'", "L2"], "F2": ["F", "F'", "F2"], "B2": ["B", "B'", "B2"],
        None: []
    }

    last_move = path[-1] if path else None
    sons = {k: v for k, v in sons.items() if k not in excluded[last_move]}
    best_son = 21
    for son in sons:
        son_cube = cube.copy()
        son_cube(son)
        sons[son] = h(son_cube)
    sorted_sons = sorted(sons, key=sons.get)
    for son in sorted_sons:
        son_cube = cube.copy()
        son_cube(son)
        result = dfs(son_cube, g+1, limit, path + [son])
        if isinstance(result, list):
            return result
        if result < best_son:
            best_son = result
    return best_son


def ida (cube):
    limit = h(cube)
    for i in range(20):
        result = dfs(cube, 0, limit, [])
        if isinstance(result, list):
            return result
        limit = result

def scrambler(cube, times):
    solution = [None] * times
    moves = ["U", "D", "R",
            "L", "F", "B",
            "U'", "D'", "R'",
            "L'", "F'", "B'",
            "U2", "D2", "R2",
            "L2", "F2", "B2"
    ]
    for i in range(times):
        movement = random.choice(moves)
        cube(movement)
        solution[i] = movement
    return (cube, solution)

if __name__ == "__main__":
    try:
        heuristic.load_state_dict(torch.load('heuristic.pth'))
        print("Pesos cargados OK")
    except FileNotFoundError:
        pass
    cube = pc.Cube()
    cube, solution = scrambler(cube, 10)
    print(solution)
    print(ida(cube))