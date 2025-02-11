import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import *
from dibs.utils import visualize_ground_truth
from dibs.models import ErdosReniDAGDistribution, BGe, ScaleFreeDAGDistribution, DenseNonlinearGaussian, LinearGaussian
from dibs.inference import MarginalDiBS, JointDiBS
from dibs.graph_utils import elwise_acyclic_constr_nograd
from jax.scipy.special import logsumexp
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".9"

ENDOGENOUS_NODES = ['Bug']

thd = 0.0
data_path = "../Data/selected"
result_path = "../Data/graph/"

releases = ['ambari-1.2.0', 'ambari-2.1.0', 'ambari-2.2.0', 'ambari-2.4.0', 'ambari-2.5.0', 'ambari-2.6.0',
            'amq-5.0.0', 'amq-5.1.0', 'amq-5.2.0', 'amq-5.4.0', 'amq-5.5.0', 'amq-5.6.0', 'amq-5.7.0', 'amq-5.8.0',
            'amq-5.9.0', 'amq-5.10.0', 'amq-5.11.0', 'amq-5.12.0', 'amq-5.14.0',
            'bookkeeper-4.0.0', 'bookkeeper-4.2.0',
            'calcite-1.6.0', 'calcite-1.8.0', 'calcite-1.11.0', 'calcite-1.13.0',
            'calcite-1.15.0', 'calcite-1.16.0', 'calcite-1.17.0',
            'cassandra-0.7.4', 'cassandra-0.8.6', 'cassandra-1.0.9', 'cassandra-1.1.6', 'cassandra-1.1.11',
            'flink-1.4.0',
            'groovy-1.0', 'groovy-1.5.5', 'groovy-1.6.0', 'groovy-1.7.3', 'groovy-1.7.6', 'groovy-1.8.1', 'groovy-1.8.7',
            'groovy-2.1.0', 'groovy-2.1.6', 'groovy-2.4.4', 'groovy-2.4.6', 'groovy-2.4.8', 'groovy-2.5.0',
            'hbase-0.94.1', 'hbase-0.94.5', 'hbase-0.98.0', 'hbase-0.98.5',
            'hive-0.14.0', 'hive-1.2.0', 'hive-2.0.0',
            'ignite-1.0.0', 'ignite-1.4.0',
            'log4j2-2.0', 'log4j2-2.1', 'log4j2-2.2', 'log4j2-2.3', 'log4j2-2.4', 'log4j2-2.5', 'log4j2-2.6', 'log4j2-2.7',
            'log4j2-2.8', 'log4j2-2.9',
            'mahout-0.3', 'mahout-0.4', 'mahout-0.5', 'mahout-0.6', 'mahout-0.7',
            'mng-3.0.0', 'mng-3.1.0', 'mng-3.2.0', 'mng-3.3.0', 'mng-3.5.0',
            'nifi-0.4.0', 'nifi-1.2.0', 'nifi-1.5.0',
            'nutch-1.1', 'nutch-1.3', 'nutch-1.4', 'nutch-1.5', 'nutch-1.6', 'nutch-1.7', 'nutch-1.8', 'nutch-1.9',
            'nutch-1.10', 'nutch-1.12', 'nutch-1.13', 'nutch-1.14',
            'storm-0.9.0', 'storm-0.9.3', 'storm-1.0.0', 'storm-1.0.3',
            'tika-0.7', 'tika-0.8', 'tika-0.9', 'tika-0.10', 'tika-1.1', 'tika-1.3', 'tika-1.5', 'tika-1.7', 'tika-1.10',
            'tika-1.13', 'tika-1.15',
            'ww-2.0.0', 'ww-2.0.5', 'ww-2.0.10', 'ww-2.1.1', 'ww-2.1.3', 'ww-2.1.7', 'ww-2.2.0', 'ww-2.2.2', 'ww-2.3.1',
            'ww-2.3.4', 'ww-2.3.10', 'ww-2.3.15', 'ww-2.3.17', 'ww-2.3.20',
            'zookeeper-3.4.6', 'zookeeper-3.5.1', 'zookeeper-3.5.2',
]

collected_df = pd.DataFrame()


def read_data(folder: str, selected_name:str) -> pd.DataFrame:
    df = pd.DataFrame()
    for dir_path, _, file_names in os.walk(folder):
        for file_name in file_names:
            if file_name == selected_name:
                file_path = os.path.join(folder, file_name)
                df = pd.read_csv(file_path)
                print(f"Read {file_name} with {df.shape[0]} rows")
    return df


def matrix_to_dgraph(matrix: np.ndarray, columns: List[str], threshold: float = 1.0) -> List[str]:
    dgraph = []
    for i in range(matrix.shape[0]):
        if matrix[i, collected_df.shape[1] - 1] > threshold:
            dgraph.append(f"{columns[i]} -> {columns[collected_df.shape[1] - 1]} :{matrix[i, collected_df.shape[1] - 1]}")
    return dgraph


def compute_expected_graph(*, dist):
    n_vars = dist.g.shape[1]

    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    assert is_dag.sum() > 0,  "No acyclic graphs found"

    particles = dist.g[is_dag, :, :]
    log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])

    expected_g = jnp.zeros_like(particles[0])
    for i in range(particles.shape[0]):
        expected_g += jnp.exp(log_weights[i]) * particles[i, :, :]

    return expected_g


def discover_barinel(release):
    selected_name = release+'_barinel.csv'
    graph_file_name = release+'_barinel_SF_graph.txt'
    rand_key = jax.random.PRNGKey(0)

    collected_df = read_data(data_path, selected_name)
    collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    collected_df.dropna(inplace=True)
    collected_df = collected_df.sample(frac=1).reset_index(drop=True)

    print(f"Collected data shape: {collected_df.shape}")
    print(f"Collected data columns: {collected_df.columns}")

    interv_df = collected_df.copy()
    for col in interv_df.columns:
        if col in ENDOGENOUS_NODES:
            interv_df[col] = 0
    for col in interv_df.columns:
        if col not in ENDOGENOUS_NODES:
            interv_df[col] = 1
    interv_mask = interv_df.values
    interv_mask[interv_mask > 0] = 1
    interv_mask = interv_mask.astype(int)
    interv_mask = jnp.array(interv_mask)

    scaler = StandardScaler()
    collected_data = scaler.fit_transform(collected_df)

    model_graph = ScaleFreeDAGDistribution(collected_data.shape[1], n_edges_per_node=2)
    model = BGe(graph_dist=model_graph)
    dibs = MarginalDiBS(x=collected_data, interv_mask=interv_mask, inference_model=model)

    rand_key, subk = jax.random.split(rand_key)
    gs = dibs.sample(key=subk, n_particles=10, steps=600, callback_every=600, callback=None)

    print(f"dibs sample is finished")

    dibs_output = dibs.get_mixture(gs)
    expected_g = compute_expected_graph(dist=dibs_output)

    visualize_ground_truth(jnp.array(expected_g))
    dgraph = matrix_to_dgraph(expected_g, collected_df.columns, threshold=thd)
    f = open(result_path+graph_file_name, 'w')
    print(len(dgraph), end='\n', file=f)
    for line in dgraph:
        print(line, end='\n', file=f)
    f.close()


def discover_op2(release):
    selected_name = release+'_op2.csv'
    graph_file_name = release+'_op2_SF_graph.txt'
    rand_key = jax.random.PRNGKey(0)

    collected_df = read_data(data_path, selected_name)
    collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    collected_df.dropna(inplace=True)
    collected_df = collected_df.sample(frac=1).reset_index(drop=True)

    print(f"Collected data shape: {collected_df.shape}")
    print(f"Collected data columns: {collected_df.columns}")

    interv_df = collected_df.copy()
    for col in interv_df.columns:
        if col in ENDOGENOUS_NODES:
            interv_df[col] = 0
    for col in interv_df.columns:
        if col not in ENDOGENOUS_NODES:
            interv_df[col] = 1
    interv_mask = interv_df.values
    interv_mask[interv_mask > 0] = 1
    interv_mask = interv_mask.astype(int)
    interv_mask = jnp.array(interv_mask)

    scaler = StandardScaler()
    collected_data = scaler.fit_transform(collected_df)

    model_graph = ScaleFreeDAGDistribution(collected_data.shape[1], n_edges_per_node=2)
    model = BGe(graph_dist=model_graph)
    dibs = MarginalDiBS(x=collected_data, interv_mask=interv_mask, inference_model=model)

    rand_key, subk = jax.random.split(rand_key)
    gs = dibs.sample(key=subk, n_particles=10, steps=600, callback_every=600, callback=None)

    print(f"dibs sample is finished")

    dibs_output = dibs.get_mixture(gs)
    expected_g = compute_expected_graph(dist=dibs_output)

    visualize_ground_truth(jnp.array(expected_g))
    dgraph = matrix_to_dgraph(expected_g, collected_df.columns, threshold=thd)
    f = open(result_path+graph_file_name, 'w')
    print(len(dgraph), end='\n', file=f)
    for line in dgraph:
        print(line, end='\n', file=f)
    f.close()


def discover_dstar(release):
    selected_name = release+'_dstar.csv'
    graph_file_name = release+'_dstar_SF_graph.txt'
    rand_key = jax.random.PRNGKey(0)

    collected_df = read_data(data_path, selected_name)
    collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    collected_df.dropna(inplace=True)
    collected_df = collected_df.sample(frac=1).reset_index(drop=True)

    print(f"Collected data shape: {collected_df.shape}")
    print(f"Collected data columns: {collected_df.columns}")

    interv_df = collected_df.copy()
    for col in interv_df.columns:
        if col in ENDOGENOUS_NODES:
            interv_df[col] = 0
    for col in interv_df.columns:
        if col not in ENDOGENOUS_NODES:
            interv_df[col] = 1
    interv_mask = interv_df.values
    interv_mask[interv_mask > 0] = 1
    interv_mask = interv_mask.astype(int)
    interv_mask = jnp.array(interv_mask)

    scaler = StandardScaler()
    collected_data = scaler.fit_transform(collected_df)

    model_graph = ScaleFreeDAGDistribution(collected_data.shape[1], n_edges_per_node=2)
    model = BGe(graph_dist=model_graph)
    dibs = MarginalDiBS(x=collected_data, interv_mask=interv_mask, inference_model=model)

    rand_key, subk = jax.random.split(rand_key)
    gs = dibs.sample(key=subk, n_particles=10, steps=600, callback_every=600, callback=None)

    print(f"dibs sample is finished")

    dibs_output = dibs.get_mixture(gs)
    expected_g = compute_expected_graph(dist=dibs_output)

    visualize_ground_truth(jnp.array(expected_g))
    dgraph = matrix_to_dgraph(expected_g, collected_df.columns, threshold=thd)
    f = open(result_path+graph_file_name, 'w')
    print(len(dgraph), end='\n', file=f)
    for line in dgraph:
        print(line, end='\n', file=f)
    f.close()


def discover_ochiai(release):
    selected_name = release+'_ochiai.csv'
    graph_file_name = release+'_ochiai_SF_graph.txt'
    rand_key = jax.random.PRNGKey(0)

    collected_df = read_data(data_path, selected_name)
    collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    collected_df.dropna(inplace=True)
    collected_df = collected_df.sample(frac=1).reset_index(drop=True)

    print(f"Collected data shape: {collected_df.shape}")
    print(f"Collected data columns: {collected_df.columns}")

    interv_df = collected_df.copy()
    for col in interv_df.columns:
        if col in ENDOGENOUS_NODES:
            interv_df[col] = 0
    for col in interv_df.columns:
        if col not in ENDOGENOUS_NODES:
            interv_df[col] = 1
    interv_mask = interv_df.values
    interv_mask[interv_mask > 0] = 1
    interv_mask = interv_mask.astype(int)
    interv_mask = jnp.array(interv_mask)

    scaler = StandardScaler()
    collected_data = scaler.fit_transform(collected_df)

    model_graph = ScaleFreeDAGDistribution(collected_data.shape[1], n_edges_per_node=2)
    model = BGe(graph_dist=model_graph)
    dibs = MarginalDiBS(x=collected_data, interv_mask=interv_mask, inference_model=model)

    rand_key, subk = jax.random.split(rand_key)
    gs = dibs.sample(key=subk, n_particles=10, steps=600, callback_every=600, callback=None)

    print(f"dibs sample is finished")

    dibs_output = dibs.get_mixture(gs)
    expected_g = compute_expected_graph(dist=dibs_output)

    visualize_ground_truth(jnp.array(expected_g))
    dgraph = matrix_to_dgraph(expected_g, collected_df.columns, threshold=thd)
    f = open(result_path+graph_file_name, 'w')
    print(len(dgraph), end='\n', file=f)
    for line in dgraph:
        print(line, end='\n', file=f)
    f.close()


def discover_tarantula(release):
    selected_name = release+'_tarantula.csv'
    graph_file_name = release+'_tarantula_SF_graph.txt'
    rand_key = jax.random.PRNGKey(0)

    collected_df = read_data(data_path, selected_name)
    collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    collected_df.dropna(inplace=True)
    collected_df = collected_df.sample(frac=1).reset_index(drop=True)

    print(f"Collected data shape: {collected_df.shape}")
    print(f"Collected data columns: {collected_df.columns}")

    interv_df = collected_df.copy()
    for col in interv_df.columns:
        if col in ENDOGENOUS_NODES:
            interv_df[col] = 0
    for col in interv_df.columns:
        if col not in ENDOGENOUS_NODES:
            interv_df[col] = 1
    interv_mask = interv_df.values
    interv_mask[interv_mask > 0] = 1
    interv_mask = interv_mask.astype(int)
    interv_mask = jnp.array(interv_mask)

    scaler = StandardScaler()
    collected_data = scaler.fit_transform(collected_df)

    model_graph = ScaleFreeDAGDistribution(collected_data.shape[1], n_edges_per_node=2)
    model = BGe(graph_dist=model_graph)
    dibs = MarginalDiBS(x=collected_data, interv_mask=interv_mask, inference_model=model)

    rand_key, subk = jax.random.split(rand_key)
    gs = dibs.sample(key=subk, n_particles=10, steps=600, callback_every=600, callback=None)

    print(f"dibs sample is finished")

    dibs_output = dibs.get_mixture(gs)
    expected_g = compute_expected_graph(dist=dibs_output)

    visualize_ground_truth(jnp.array(expected_g))
    dgraph = matrix_to_dgraph(expected_g, collected_df.columns, threshold=thd)
    f = open(result_path+graph_file_name, 'w')
    print(len(dgraph), end='\n', file=f)
    for line in dgraph:
        print(line, end='\n', file=f)
    f.close()


for release in releases:
    discover_barinel(release)
    discover_op2(release)
    discover_dstar(release)
    discover_ochiai(release)
    discover_tarantula(release)
