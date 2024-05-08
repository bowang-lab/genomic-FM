from ..dataloader import data_wrapper as data_wrapper
from ..model_wrapper.base_model import BaseModel
import yaml
import faiss
import numpy as np
from ..dataloader.save_as_np import get_cache
from ..dataloader.memmap_dataset import MemMapDataset
from tqdm import tqdm
import torch


class VectorLoader:
    def __init__(self, dataset="clinvar_CLNSIG_hyena-tiny", config_path="/jmain02/home/J2AD015/axf03/zxl79-axf03/repository/genomic-FM-run-exp/genomic-FM/configs/vector_index.yaml",from_cache=False, cache_dir="root/data/npy_output"):
        # Load configuration file
        with open(config_path, 'r') as f:
            info = yaml.load(f, Loader=yaml.FullLoader)
        info = info[dataset]
        pca_components = info.pop('pca_components')
        self.pca_components = pca_components
        # Create model
        num_records = None
        if 'num_records' in info and 'all_records' in info:
            num_records = info.pop('num_records')
        self.model = BaseModel(model_initiator_name=info.pop('model_initiator_name'))
        self.database_name = info.pop('task')

        # Perform Embedding with mini-batch
        mini_batch = 64
        vectors = None
        labels = []

        if from_cache:
            seq1_path, seq2_path, annot_path, label_path = get_cache(dataset, cache_dir)
            memmap_data = MemMapDataset(path_seq1=seq1_path,
                                        path_seq2=seq2_path,
                                        seq_shape=(info['Seq_length'], pca_components),
                                        chunk_size=info['Seq_length'],
                                        annotation_paths=annot_path,
                                        label_paths=label_path)
            loader = torch.utils.data.DataLoader(memmap_data, batch_size=mini_batch, shuffle=True, num_workers=18, pin_memory=True)
            if num_records is None:
                num_batches = len(loader)
            else:
                num_batches = num_records // mini_batch
            if num_records % mini_batch != 0:
                num_batches += 1
            for i, batch in enumerate(tqdm(loader, desc="Processing batches", total=num_batches, unit="batch")):
                if i == num_batches:
                    break
                batch_vectors = batch[0][1] - batch[0][0]
                batch_vectors = batch_vectors.reshape(batch_vectors.shape[0], -1)
                if vectors is None:
                    vectors = batch_vectors
                else:
                    vectors = np.vstack((vectors, batch_vectors))
                labels.extend(batch[1])
        else:
            cls = getattr(data_wrapper, info.pop('class'))
            if num_records is not None:
                DATA = cls(num_records=num_records, all_records=info.pop('all_records'))
            else:
                DATA = cls()

            self.data = DATA.get_data(**info)
            for i in range(0, len(self.data), mini_batch):
                vectors_batch, labels_batch = self.model.cache_embed_delta(self.data[i:i+mini_batch], pca_components)
                # flatten each vector in the vector batch
                vectors_batch = vectors_batch.reshape(vectors_batch.shape[0], -1)
                if vectors is None:
                    vectors = vectors_batch
                else:
                    vectors = np.vstack((vectors, vectors_batch))
                labels.extend(labels_batch)
        self.vectors = vectors
        self.labels = labels
        self.init_faiss_index()

    def encode_to_vector(self, data):
        """Returns the difference of the embeddings of the two sequences in the input data.
           and the corresponding labels."""
        return self.model.cache_embed_delta(data, self.pca_components)


    def init_faiss_index(self):
        dimension = len(self.vectors[0])  # Assuming all vectors are of same dimension
        self.index = faiss.IndexFlatL2(dimension)  # Using L2 distance metric
        self.index.add(np.array(self.vectors))  # Adding vectors to the index


    def query_vectors(self, query_vector, k=4):
        # Ensure the query vector is in the correct shape (1, dimension)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        # Perform the search
        distances, indices = self.index.search(query_vector, k)
        # Retrieve the corresponding labels
        result_labels = [self.labels[i] for i in indices[0]]
        return distances[0], result_labels, indices


    def perform_clustering(self, n_clusters=5, n_iter=20, verbose=True):
        kmeans = faiss.Kmeans(d=len(self.vectors[0]), k=n_clusters, niter=n_iter, verbose=verbose)
        kmeans.train(np.array(self.vectors))
        # Assign clusters
        distances, assignments = kmeans.index.search(np.array(self.vectors), 1)
        return assignments.flatten(), distances.flatten()
