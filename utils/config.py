import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    archive_rows: int
    archive_cols: int
    gen: int
    population_size: int
    num_inputs: int
    num_outputs: int
    num_hidden: int
    activations: list
    output_activation: str
    mutation_rate: float
    p_sigma: float
    split_edge_rate: float
    add_edge_rate: float
    default_activation: str
    delete_node_rate: float
    delete_edge_rate: float
    mutation_activation_rate: float
    is_cvt: bool
    cvt_file: str
    cell_size: int
    bias_mean: float
    bias_std: float
    weight_mean: float
    weight_std: float

    @staticmethod
    def load(config_file):
        with open(config_file, 'r') as file:
            config_data = json.load(file)
        return Config(
            archive_rows=config_data['archive_rows'],
            archive_cols=config_data['archive_cols'],
            gen=config_data['gen'],
            population_size=config_data['population_size'],
            num_inputs=config_data['num_inputs'],
            num_outputs=config_data['num_outputs'],
            num_hidden=config_data['num_hidden'],
            activations=config_data['activations'],
            output_activation=config_data['output_activation'],
            mutation_rate=config_data['mutation_rate'],
            p_sigma=config_data['p_sigma'],
            split_edge_rate=config_data['split_edge_rate'],
            add_edge_rate=config_data['add_edge_rate'],
            default_activation=config_data['default_activation'],
            delete_node_rate=config_data['delete_node_rate'],
            delete_edge_rate=config_data['delete_edge_rate'],
            mutation_activation_rate=config_data['mutation_activation_rate'],
            is_cvt=config_data['is_cvt'],
            cvt_file=config_data['cvt_file'],
            cell_size=config_data['cell_size'],
            bias_mean=config_data['bias_mean'],
            bias_std=config_data['bias_std'],
            weight_mean=config_data['weight_mean'],
            weight_std=config_data['weight_std'],
        )
