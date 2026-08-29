import os
from .contriever_src.contriever import Contriever
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import json
import numpy as np
import pickle
import random
import torch
from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer
from loguru import logger
import os

import sys
 
import time

model_code_to_qmodel_name = {
    "contriever": "/T20200133/Models/facebook/contriever",
    "contriever-msmarco": "/T20200133/Models/facebook/contriever-msmarco",
    "ance": "/T20200133/Models/sentence-transformers/msmarco-roberta-base-ance-firstp"
}

model_code_to_cmodel_name = {
    "contriever": "/T20200133/Models/facebook/contriever",
    "contriever-msmarco": "/T20200133/Models/facebook/contriever-msmarco",
    "ance": "/T20200133/Models/sentence-transformers/msmarco-roberta-base-ance-firstp"
}


def load_cached_data(cache_file, load_function, *args, **kwargs):
    if os.path.exists(cache_file):
        logger.info(f"Cache file {cache_file} exists. Loading data...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        logger.info(f"Cache file {cache_file} does not exist. Generating data...")
        data = load_function(*args, **kwargs)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        return data
    


def setup_experiment_logging(experiment_name=None, log_dir='logs'):
    """
    Configure logging for experiments with both console and file output.
    
    Args:
        experiment_name: Name of the experiment for the log file
        log_dir: Directory to store log files
    """
    # Remove any existing handlers
    logger.remove()
    
    # Add console handler with a simple format
    logger.add(sys.stderr, format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")
    
    # Add file handler if experiment_name is provided
    if experiment_name:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="INFO"
        )
        
    return logger 

def contriever_get_emb(model, input):
    return model(**input)

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

def load_models(model_code):
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if 'contriever' in model_code:
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif 'ance' in model_code:
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError
    # model: for query embedding
    # c_model: for context embedding
    return model, c_model, tokenizer, get_emb

def load_beir_datasets(dataset_name, split):
    assert dataset_name in ['nq', 'msmarco', 'hotpotqa']
    if dataset_name == 'msmarco': split = 'train'
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, dataset_name)
    if not os.path.exists(data_path):
        print(f"Downloading {dataset_name} ...")
        print("out_dir: ", out_dir)
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        split = 'train'
    corpus, queries, qrels = data.load(split=split)

    return corpus, queries, qrels

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        
def save_outputs(outputs, dir, file_name):
    json_dict = json.dumps(outputs, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f'data_cache/outputs/{dir}'):
        os.makedirs(f'data_cache/outputs/{dir}', exist_ok=True)
    with open(os.path.join(f'data_cache/outputs/{dir}', f'{file_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f, indent=4)

def save_results(results, dir, file_name="debug"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f'results/query_results/{dir}'):
        os.makedirs(f'results/query_results/{dir}', exist_ok=True)
    with open(os.path.join(f'results/query_results/{dir}', f'{file_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f, indent=4)

def load_results(file_name):
    with open(os.path.join('results', file_name)) as file:
        results = json.load(file)
    return results

def save_json(results, file_path="debug.json"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f, indent=4)

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_str(s):
    try:
        s=str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s=s.strip()
    if len(s)>1 and s[-1] == ".":
        s=s[:-1]
    return s.lower()

def f1_score(precision, recall):
    f1_scores = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    return f1_scores

class LoguruProgress:
    def __init__(self, iterable=None, desc=None, total=None, **kwargs):
        self.iterable = iterable
        self.desc = desc
        self.total = len(iterable) if iterable is not None else total
        self.n = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        logger.info(f"Starting {desc}: 0/{self.total}")

    def update(self, n=1):
        self.n += n
        current_time = time.time()
        # Log every second or at completion
        if current_time - self.last_log_time > 1 or self.n >= self.total:
            elapsed = current_time - self.start_time
            rate = self.n / elapsed if elapsed > 0 else 0
            logger.info(f"{self.desc}: {self.n}/{self.total} "
                       f"[{elapsed:.1f}s elapsed, {rate:.1f} it/s]")
            self.last_log_time = current_time

    def __iter__(self):
        if self.iterable is None:
            raise ValueError("Iterable not provided")
        for obj in self.iterable:
            yield obj
            self.update(1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            elapsed = time.time() - self.start_time
            rate = self.n / elapsed if elapsed > 0 else 0
            logger.info(f"Completed {self.desc}: {self.n}/{self.total} "
                       f"[{elapsed:.1f}s elapsed, {rate:.1f} it/s]")

def progress_bar(iterable=None, desc=None, total=None, **kwargs):
    """
    A wrapper function that returns either LoguruProgress or tqdm based on whether we want
    logging output or standard tqdm output
    """
    return LoguruProgress(iterable=iterable, desc=desc, total=total, **kwargs)