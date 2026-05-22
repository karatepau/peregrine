import sys
import pycuber as pc
import random
import pickle

def generator (size):
    cubes = [None] * size
    solutions = [None] * size
    FACES = ["U", "D", "F", "B", "R", "L"]
    COLOR_TO_ONE_HOT = {
        "white":  [1, 0, 0, 0, 0, 0],
        "yellow": [0, 1, 0, 0, 0, 0],
        "red":    [0, 0, 1, 0, 0, 0],
        "orange": [0, 0, 0, 1, 0, 0],
        "green":  [0, 0, 0, 0, 1, 0],
        "blue":   [0, 0, 0, 0, 0, 1]
    }
    ALLOWED_NEXT_FACES = {
        "U": ["R", "L", "F", "B"],
        "D": ["R", "L", "F", "B"],
        "R": ["U", "D", "F", "B"],
        "L": ["U", "D", "F", "B"],
        "F": ["U", "D", "R", "L"],
        "B": ["U", "D", "R", "L"],
        None: ["U", "D", "R", "L", "F", "B"]
    }
    MODIFIERS = ["", "'", "2"]

    c = 0
    for n in range(1, 21):
        for _ in range(size // 20):
            print(f"\r{c}/{size}", end="", flush=True)
            cube = pc.Cube()
            current_face = None
            normalized_cube = []
            for i in range(n):
                next_face = random.choice(ALLOWED_NEXT_FACES[current_face])
                modifier = random.choice(MODIFIERS)
                random_move = next_face + modifier
                cube(random_move)
                current_face = next_face
            for j, face in enumerate(FACES):
                for row in cube.get_face(face):
                    for sticker in row:
                        normalized_cube.extend(COLOR_TO_ONE_HOT[sticker.colour])
            cubes[c] = normalized_cube
            solutions[c] = n
            c += 1
    return (cubes, solutions)

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    print(f"Generando {size} cubos...")
    cubes, solutions = generator(size)
    with open('dataset.pkl', 'wb') as f:
        pickle.dump((cubes, solutions), f)
    print("Dataset guardado en dataset.pkl")
