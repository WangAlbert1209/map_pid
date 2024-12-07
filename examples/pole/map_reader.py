import pickle

from MAPELITE.map import Archive

with open("./map_archive.pkl", "rb") as f:
    archive = pickle.load(f)
    map = Archive(0, 0,is_cvt=True,cvt_file="../../MAPELITE/centroids_1000_dim2.dat")
    map.archive = archive
    map.display_archive()
    # for b, g in archive.items():
    #     if g.fitness>60:
    #         g.draw()
