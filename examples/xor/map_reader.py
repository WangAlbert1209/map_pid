import pickle

with open("./map_archive.pkl", "rb") as f:
    archive = pickle.load(f)
    for b, g in archive.items():
        if abs(g.fitness) < 1e-20:
            print(g.fitness)
            g.draw()
