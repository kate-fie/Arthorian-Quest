__all__ = ['QueryArthor']

import requests
import warnings
import pandas as pd
from typing import List, Optional, Dict, Union
from rdkit import Chem
from rdkit.Chem import PandasTools, Draw, AllChem
from pathlib import Path
import logging
from time import sleep
from tqdm import tqdm
import json
from datetime import datetime
import os
import pickle


class QueryArthor:
    """
    Query class for Arthorian Quest.
    It queries arthor.docking.org via ``.retrieve`` or ``.batch_retrieve``

    See https://arthor.docking.org/api.html for the API endpoints used

    .. code-block:: python
        Query().retrieve('[CH3]-[CH2X4]', ['BB-50-22Q1'])

        # For batch queries:
        Query().batch_retrieve(['[CH3]-[CH2X4]', '[OH]-[CH2]'], ['BB-50-22Q1'])
    """

    enamine_dbs = ['BB-ForSale-22Q1', 'MADE-BB-23Q1-770M', 'REAL-Database-22Q1']

    def __init__(self, base_url: str = 'https://arthor.docking.org/',
                 cache_dir: Optional[Union[str, Path]] = None):
        self.base_url = base_url
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)

    @property
    def dbs(self):
        return pd.DataFrame(requests.get(self.base_url + 'dt/data').json())

    def create_empty_metadata_row(self, query: str, query_inchi: str, data: dict, length: int,
                                  dbname: str) -> pd.DataFrame:
        """Creates a single row DataFrame with metadata when no matches are found"""
        try:
            arthor_source = data['arthor.source']
        except KeyError:
            arthor_source = dbname
        return pd.DataFrame([{
            'arthor.rank': None,
            'arthor.index': None,
            'smiles': None,
            'identifier': None,
            'arthor.source': arthor_source,
            'recordsTotal': data['recordsTotal'],
            'recordsFiltered': data['recordsFiltered'],
            'hasMore': data['hasMore'],
            'query': query,
            'query_inchi': query_inchi,
            'query_length': 0,
            'N_RB': None,
            'N_HA': None,
            'mol': None
        }])

    def retrieve(self, query: str, dbnames: List[str], search_type='Substructure', length=1_000):
        """
        Returns a dataframe of the results of the query,
        with fields:

        * N_RB: number of rotatable bonds
        * N_HA: number of heavy atoms

        :param query: SMARTS query
        :param dbnames: list of names (see self.dbs)
        :return:
        """
        dbname: str = ','.join(dbnames)
        query_inchi: Optional[str] = None
        if isinstance(query, Chem.Mol):
            query = Chem.MolToSmarts(query)
            query_inchi = Chem.MolToInchiKey(query)
        if isinstance(query, str) and query_inchi is None:
            query_inchi = Chem.MolToInchiKey(Chem.MolFromSmiles(query))

        response: requests.Response = requests.get(self.base_url + f'/dt/{dbname}/search',
                                                   dict(query=query,
                                                        type=search_type,
                                                        length=length)
                                                   )

        if response.status_code == 503:
            raise ConnectionError('Arthor unavailable. cf. https://arthor.docking.org/')

        response.raise_for_status()
        data: dict = response.json()

        if data.get("message", '') == "SMARTS query is always false!":
            warnings.warn(f"SMARTS query {query} is always false")
            return self.create_empty_metadata_row(query=query,
                                                  query_inchi=query_inchi,
                                                  data=data,
                                                  length=length,
                                                  dbname=dbname)
        if data.get('warning', ''):
            warnings.warn(data['warning'])
        if not data.get('recordsTotal', False):
            warnings.warn(f"SMARTS query {query} returned no matches")
            return self.create_empty_metadata_row(query=query,
                                                  query_inchi=query_inchi,
                                                  data=data,
                                                  length=length,
                                                  dbname=dbname)

        matches = pd.DataFrame(data['data'],
                               columns=['arthor.rank', 'arthor.index', 'smiles', 'identifier', 'arthor.source'])

        if len(matches) == 0:  # empty
            return self.create_empty_metadata_row(query=query,
                                                  query_inchi=query_inchi,
                                                  data=data,
                                                  length=length,
                                                  dbname=dbname)

        matches['arthor.source'] = matches['arthor.source'].apply(lambda x: x.replace('\t', ''))
        # add metadata from query
        matches['recordsTotal'] = data['recordsTotal']
        matches['recordsFiltered'] = data['recordsFiltered']
        matches['hasMore'] = data['hasMore']
        matches['query'] = query
        matches['query_inchi'] = query_inchi
        matches['query_length'] = length
        matches = matches.drop_duplicates('arthor.index')
        PandasTools.AddMoleculeColumnToFrame(matches, 'smiles', 'mol', includeFingerprints=True)
        matches = matches.loc[~matches.mol.isnull()]
        matches['N_RB'] = matches.mol.apply(AllChem.CalcNumRotatableBonds)
        matches['N_HA'] = matches.mol.apply(AllChem.CalcNumHeavyAtoms)
        return matches.sort_values('N_HA').reset_index(drop=True)

    def batch_retrieve(self,
                       queries: List[str],
                       dbnames: List[str],
                       search_type: str = 'Substructure',
                       length: int = 10_000,
                       sleep_time: float = 5.0,
                       continue_on_error: bool = True,
                       batch_size: int = 10) -> str:
        """
        Perform batch retrieval of multiple queries with progress tracking and error handling.

        Parameters
        ----------
        queries : List[str]
            List of SMARTS or SMILES queries
        dbnames : List[str]
            List of database names to search
        search_type : str, optional
            Type of search ('SMARTS' or 'Substructure' or 'Similarity'), by default 'Substructure'
        length : int, optional
            Maximum number of results per database per query, by default 10_000.
        sleep_time : float, optional
            Time to wait between queries in seconds, by default 1.0
        continue_on_error : bool, optional
            Whether to continue processing on error, by default True
        batch_size : int, optional
            Number of queries to save in intermittant dataframes, by default 10

        Returns
        -------
        str
            Path to the cache directory where batched dataframes are saved

        Examples
        --------
        >>> queries = ['[CH3]-[CH2X4]', '[OH]-[CH2]']
        >>> qa = QueryArthor(cache_dir='query_cache')
        >>> results: pd.DataFrame = qa.batch_retrieve(queries, ['BB-50-22Q1'])
        """
        logging.basicConfig(filename=os.path.join(self.cache_dir if self.cache_dir else Path.cwd(),
                                                  f"arthorian_quest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        if not self.cache_dir:
            logging.warning("No cache directory set. Will use current directory for logging and saving.")
            self.cache_dir = Path.cwd()
        logging.info(f"Cache directory: {self.cache_dir}")

        # Initialize or load progress tracking
        progress_file: str | None = os.path.join(self.cache_dir, "batch_progress.json") if self.cache_dir else None
        logging.info(f"Progress file: {progress_file}")
        completed_batches = {}
        if progress_file and os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                completed_batches = json.load(f)

        # Check queries is a list
        if not all(isinstance(q, str) for q in queries):
            logging.error("Queries must be a list of strings!")
            logging.error(f"Queries provided: {queries}")
            raise TypeError

        # Check dbnames is a list
        if not all(isinstance(db, str) for db in dbnames):
            logging.error("dbnames must be a list of strings!")
            logging.error(f"dbnames provided: {dbnames}")
            raise TypeError

        # Create batches of queries
        query_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]

        for batch_idx, query_batch in enumerate(tqdm(query_batches, desc="Processing batches")):
            batch_id = f"batch_{batch_idx}"
            batch_file = os.path.join(self.cache_dir, f"batch_{batch_idx}.pkl.gz")
            query_inchis = [Chem.MolToInchiKey(Chem.MolFromSmiles(q)) for q in query_batch]
            assert len(query_inchis) == len(query_batch), "Failed to convert queries to InChI keys"

            # Skip if batch is already completed
            if batch_id in completed_batches and completed_batches[batch_id]:
                batch_df = pd.read_pickle(batch_file)
                if all(inchi in batch_df.query_inchi.values for inchi in query_inchis):
                    logging.info(f"Skipping completed batch {batch_idx}")
                    continue
                logging.info(f"Batch {batch_idx} has missing queries, reprocessing")

            batch_results = []
            try:
                for query in query_batch:
                    for dbname in dbnames:
                        df = self.retrieve(query, [dbname], search_type, length)
                        if df is not None and not df.empty:
                            batch_results.append(df)
                            logging.info(f"Successfully processed query: {query} {dbname} ({len(df)} results)")
                    sleep(sleep_time)

                if len(batch_results) == len(query_batch) * len(dbnames):
                    batch_df = pd.concat(batch_results, ignore_index=True)
                    batch_df.to_pickle(batch_file)

                    # Update progress only after successful batch save
                    completed_batches[batch_id] = True
                    with open(progress_file, 'w') as f:
                        json.dump(completed_batches, f)
                    logging.info(f"Completed and saved batch {batch_idx}")

            except Exception as e:
                error_msg = f"Error processing batch {batch_idx}: {str(e)}"
                logging.error(error_msg)
                if progress_file:
                    completed_batches[batch_id] = False
                    with open(progress_file, 'w') as f:
                        json.dump(completed, f)
                if not continue_on_error:
                    raise Exception(error_msg)

        return self.cache_dir

    def get_batch_statistics(self) -> Dict:
        """
        Get statistics about batch processing progress.
        Only available if cache_dir was set.

        Returns
        -------
        Dict
            Dictionary containing progress statistics
        """
        if not self.cache_dir:
            return {"error": "No cache directory set"}

        progress_file = self.cache_dir / "batch_progress.json"
        if not progress_file.exists():
            return {"error": "No batch processing history found"}

        with open(progress_file, 'r') as f:
            completed = json.load(f)

        return {
            "total_processed": len(completed),
            "successful": sum(1 for v in completed.values() if v),
            "failed": sum(1 for v in completed.values() if not v)
        }