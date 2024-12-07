import neat
class CustomGenome(neat.DefaultGenome):
    """
    A custom genome class that extends DefaultGenome with overridden methods.
    """
    def __init__(self, key):
        super().__init__(key)
        # Add any additional initialization here if needed
    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in parent1.connections.items():
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)
