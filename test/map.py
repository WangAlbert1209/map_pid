import numpy as np

from MAPELITE.map import Archive


# Mock Genome class for testing
class Genome:
    def __init__(self, behavior, fitness):
        self.behavior = behavior  # array representing the behavior descriptor
        self.fitness = fitness  # float representing the fitness of the genome
        self.niche = None  # will be set when added to the archive


def test_archive_initialization():
    # Initialize Archive with and without CVT
    archive_grid = Archive(10, 10, is_cvt=False)
    assert archive_grid.kdt is not None, "KDTree should be initialized for grid-based Archive."

    # For CVT archive, provide a mock centroid file
    np.savetxt("cvt_test_file.txt", np.random.rand(50, 2))  # create a mock CVT file
    archive_cvt = Archive(10, 10, is_cvt=True, cvt_file="cvt_test_file.txt")
    assert archive_cvt.kdt is not None, "KDTree should be initialized for CVT Archive."
    print("Archive initialization tests passed.")


def test_find_position():
    # Initialize a grid-based archive
    archive = Archive(10, 10, is_cvt=False)
    behavior = np.array([0.5, 0.5])

    position = archive.find_position(behavior)
    print("postion is :", position)
    assert position is not None, "Position should be found for valid behavior descriptor."
    print("find_position test passed.")


def test_add_to_archive():
    # Initialize Archive
    archive = Archive(10, 10, is_cvt=True, cvt_file="../MAPELITE/centroids_1000_dim2.dat")

    # Create mock genomes and add to archive
    genome1 = Genome(behavior=np.array([0.2, 0.2]), fitness=5.0)
    genome2 = Genome(behavior=np.array([0.2, 0.2]), fitness=10.0)  # higher fitness, should replace genome1
    genome3 = Genome(behavior=np.array([0.5, 0.2]), fitness=5.0)
    archive.add_to_archive(genome1)
    archive.add_to_archive(genome3)
    assert archive.find_position(genome1.behavior) in archive.archive, "Genome should be added to archive."

    archive.add_to_archive(genome2)
    archive.display_archive()
    assert archive.archive[archive.find_position(genome2.behavior)].fitness == 10.0, \
        "Genome with higher fitness should replace the existing one."
    print("add_to_archive test passed.")


def test_cal_qd_score():
    # Initialize Archive and add some genomes
    archive = Archive(10, 10, is_cvt=True,cvt_file="../MAPELITE/centroids_1000_dim2.dat")
    genome1 = Genome(behavior=np.array([0.2, 0.2]), fitness=5.0)
    genome2 = Genome(behavior=np.array([0.8, 0.8]), fitness=15.0)

    archive.add_to_archive(genome1)
    archive.add_to_archive(genome2)

    filling_rate, mean_fit = archive.cal_qd_score()
    print(filling_rate, mean_fit)
    assert 0 <= filling_rate <= 1, "Filling rate should be between 0 and 1."
    assert mean_fit >= 0, "Mean fitness should be non-negative."
    print("cal_qd_score test passed.")


def test_display_archive():
    # Initialize Archive and add some genomes
    archive = Archive(10, 10, is_cvt=True,cvt_file="../MAPELITE/centroids_1000_dim2.dat")
    genome1 = Genome(behavior=np.array([0.2, 0.2]), fitness=5.0)
    genome2 = Genome(behavior=np.array([0.8, 0.8]), fitness=15.0)

    archive.add_to_archive(genome1)
    archive.add_to_archive(genome2)

    # Display archive and check visually
    print("Displaying archive for visual verification:")
    archive.display_archive()


# Run tests
# test_archive_initialization()
# test_find_position()
# test_add_to_archive()
# test_cal_qd_score()
test_display_archive()
