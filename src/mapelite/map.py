import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import KDTree


class Archive:
    """
    Manage archive for storing genomes based on behavior descriptors.
    """

    def __init__(self, rows, cols, is_cvt=False, cvt_file=None, cell_size=1):
        self.rows = rows
        self.cols = cols
        self.is_cvt = is_cvt
        self.cvt_file = cvt_file
        self.cell_size = cell_size
        if self.cell_size > 1:
            print("Multi-Objectives Map Elites Algorithm")
        elif self.cell_size == 1:
            print("CVT Map Elites Algorithm")
        # 先初始化kdtree，用于查找在archive中的position
        if is_cvt:
            assert self.cvt_file is not None, "give the centroid array"
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_dir, self.cvt_file)
            print(f"尝试加载文件: {full_path}")
            centroid = np.loadtxt(full_path)
            
            self.kdt = KDTree(centroid, leaf_size=30, metric='euclidean')
        else:
            x_centered = np.linspace(0.5 / rows, 1 - 0.5 / rows, rows)
            y_centered = np.linspace(0.5 / cols, 1 - 0.5 / cols, cols)
            xv_centered, yv_centered = np.meshgrid(x_centered, y_centered, indexing="ij")
            centroid = np.column_stack([xv_centered.ravel(), yv_centered.ravel()])
            # Initialize KDTree for the grid
            self.kdt = KDTree(centroid, leaf_size=30, metric='euclidean')
        self.archive = {}

    def make_hashable(self, array):
        return tuple(map(float, array))

    def find_position(self, b):
        niche_index = self.kdt.query([b], k=1)[1][0][0]
        niche = self.kdt.data[niche_index]
        position = self.make_hashable(niche)
        return position

    def add_to_archive(self, genome):
        position = self.find_position(genome.behavior)

        # moqd
        if self.cell_size > 1:
            # mechanism of multi-objective task
            # 1. remove  individuals dominated by candidate
            # 2. remove  candidate if it was dominated
            self.moqd_add(position, genome)
        # standard single obj map elite
        else:
            if position not in self.archive or genome.fitness > self.archive[position].fitness:
                self.archive[position] = genome
                genome.niche = position

    def moqd_add(self, position, genome):
        """
        Add `genome` to the archive at the specified `position` while maintaining
        the Pareto front.
        """
        if position not in self.archive:
            # Initialize the niche with the new genome
            self.archive[position] = [genome]
            genome.niche = position
        else:
            # Update the Pareto front for the existing niche
            self.archive[position] = self.update_pareto_front(genome, self.archive[position])
            genome.niche = position

    def dominates(self, individual1, individual2):
        """
        Check if `individual1` dominates `individual2`.
        """
        fitness1 = individual1.fitness
        fitness2 = individual2.fitness

        assert len(fitness1) == len(fitness2), "Fitness dimensions must match"

        strictly_better = False
        for f1, f2 in zip(fitness1, fitness2):
            if f1 < f2:  # Worse in any objective
                return False
            if f1 > f2:  # Strictly better in at least one objective
                strictly_better = True

        return strictly_better

    def generate_dominance_index(self, candidate, front):
        """
        Generate indices of genomes in `front` that are dominated by `candidate`
        and check if `candidate` is dominated by any genome in `front`.
        """
        dominated_indices = []
        candidate_dominated = False

        for i, genome in enumerate(front):
            if self.dominates(candidate, genome):
                dominated_indices.append(i)  # Mark as dominated by candidate
            elif self.dominates(genome, candidate):
                candidate_dominated = True  # Candidate is dominated by this genome
                break

        return dominated_indices, candidate_dominated

    def update_pareto_front(self, candidate, front):
        """
        Update the Pareto front by adding `candidate` and removing dominated solutions.
        """
        # Generate dominance indices
        dominated_indices, candidate_dominated = self.generate_dominance_index(candidate, front)

        # If the candidate is dominated, it cannot join the front
        if candidate_dominated:
            return front

        # Remove dominated genomes and add the candidate
        updated_front = [genome for i, genome in enumerate(front) if i not in dominated_indices]
        updated_front.append(candidate)

        #  cell_size retained by Queue rather than reserve the age information in genome.
        if len(updated_front) > self.cell_size:
            updated_front.pop(0)  # Remove the first (earliest) individual

        return updated_front

    def cal_qd_score(self):
        centroid_num = len(self.kdt.data)
        print(centroid_num)
        filling_rate = len(self.archive) / centroid_num
        fit_list = [genome.fitness for genome in self.archive.values()]
        mean_fit = sum(fit_list) / centroid_num
        return filling_rate, mean_fit

    def cal_hypervolume(self):
        pass

    def visualize_map(self):
        if self.cell_size > 1:
            self.display_moqd()
        else:
            self.display_archive()

    def plot_cell_scatter(self, ax, data, label, cell_color):
        """Plot scatter plot for a single cell with consistent color for all points."""
        # Use the same color for all points in the cell
        ax.scatter(data[:, 0], data[:, 1], label=label, color=cell_color, s=20)

    def display_moqd(self):
        """Display all Pareto fronts and behavior plots for each cell in one figure."""
        # Generate random colors for each cell before plotting
        cell_colors = {key: np.random.rand(3, ) for key in self.archive.keys()}

        # Plot Pareto fronts
        fig, ax = plt.subplots(figsize=(10, 10))

        for idx, ((x, y), cell_genomes) in enumerate(self.archive.items()):
            # Extract fitness values for all genomes in the cell
            fitnesses = np.array([genome.fitness for genome in cell_genomes])

            # Retrieve the color for this cell
            cell_color = cell_colors[(x, y)]

            # Plot Pareto front for the current cell
            self.plot_cell_scatter(ax, fitnesses, f"Cell ({x:.2f}, {y:.2f})", cell_color)

        ax.set_title("Multi-Objective Pareto Fronts in Archive")
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        plt.tight_layout()
        plt.show()

        # Plot behavior space
        fig, ax = plt.subplots(figsize=(10, 10))

        # Extract centroids for plotting
        centroids = self.kdt.data
        # ax.scatter(centroids[:, 0], centroids[:, 1], c='black', s=50, label="Centroids", marker='x')

        for idx, ((x, y), cell_genomes) in enumerate(self.archive.items()):
            # Extract behavior coordinates for all genomes in the cell
            behaviors = np.array([genome.behavior for genome in cell_genomes])

            # Retrieve the color for this cell
            cell_color = cell_colors[(x, y)]

            # Plot behavior space for the current cell
            self.plot_cell_scatter(ax, behaviors, f"Cell ({x:.2f}, {y:.2f})", cell_color)

        # Plot Voronoi diagram based on centroids
        vor = Voronoi(centroids)
        voronoi_plot_2d(vor, ax=ax, show_vertices=True, line_colors='black', line_width=1.5, line_alpha=0.6)

        ax.set_title("Behavior Space Scatter Plot (CVT) with Voronoi")
        ax.set_xlabel("Behavior Dimension 1")
        ax.set_ylabel("Behavior Dimension 2")
        plt.tight_layout()
        plt.show()

    def display_archive(self):
        # Prepare data for plotting
        x_coords, y_coords, fitness_values = zip(*[(x, y, genome.fitness) for (x, y), genome in self.archive.items()])
        min_fit, max_fit = min(fitness_values), max(fitness_values)
        norm = Normalize(vmin=min_fit, vmax=max_fit)
        cmap = plt.get_cmap("plasma")

        # Determine best and worst genomes
        best_genome = max(self.archive.values(), key=lambda g: g.fitness, default=None)
        worst_genome = min(self.archive.values(), key=lambda g: g.fitness, default=None)

        if self.is_cvt:
            # Generate Voronoi diagram based on centroids from KDTree
            centroids = self.kdt.data
            vor = Voronoi(centroids)

            # Plot Voronoi regions colored by genome fitness
            fig, ax = plt.subplots(figsize=(8, 8))
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=1.5, line_alpha=0.6,
                            point_size=2)

            for (x, y), genome in self.archive.items():
                idx = np.where((centroids == np.array([x, y])).all(axis=1))[0][0]
                region = vor.regions[vor.point_region[idx]]
                if -1 not in region:  # Skip unbounded regions
                    polygon = [vor.vertices[i] for i in region]
                    ax.fill(*zip(*polygon), color=cmap(norm(genome.fitness)))

            # 修复 colorbar 的问题
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            plt.colorbar(sm, ax=ax, label="Fitness")  # 添加 ax 参数
            ax.set_title("Voronoi Diagram of Archive")
        else:
            # Scatter plot for grid-based visualization
            plt.figure(figsize=(8, 8))
            sc = plt.scatter(x_coords, y_coords, c=fitness_values, cmap='plasma', s=10)
            sc.set_clim(min_fit, max_fit)
            plt.colorbar(sc, label="Fitness")
            plt.title("Archive Fitness Scatter Plot")

        # Common plot adjustments

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(False)
        # plt.show(block=False)
        plt.savefig('./pole_map.png')
        plt.close()
        return min_fit, max_fit, best_genome, worst_genome
