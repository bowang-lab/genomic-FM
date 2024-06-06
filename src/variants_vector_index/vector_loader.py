from ..dataloader import data_wrapper as data_wrapper
from ..model_wrapper.base_model import BaseModel
import yaml
import faiss
import numpy as np
from ..dataloader.save_as_np import get_cache_delta
from ..dataloader.memmap_dataset_delta import MemMapDatasetDelta
from tqdm import tqdm
import torch
from ..model_wrapper.pl_model_delta import MyLightningModuleDelta
from ..model_wrapper.cnn_head import CNN_Head

class VectorLoader:
    def __init__(self, dataset="clinvar_CLNSIG_hyena-tiny",
                config_path="/jmain02/home/J2AD015/axf03/zxl79-axf03/repository/genomic-FM-run-exp/genomic-FM/configs/vector_index.yaml",
                from_cache=False, cache_dir="root/data/npy_output_delta", checkpoint=None, random=False):
        # Load configuration file
        with open(config_path, 'r') as f:
            info = yaml.load(f, Loader=yaml.FullLoader)
        info = info[dataset]
        pca_components = info.pop('pca_components')
        self.pca_components = pca_components
        final_pca_components = info.pop('final_pca_components')
        # Create model
        num_records = None
        if 'num_records' in info and 'all_records' in info:
            num_records = info.pop('num_records')
        self.model = BaseModel(model_initiator_name=info.pop('model_initiator_name')) if checkpoint is None else CNN_Head(model_initiator_name=info.pop('model_initiator_name'),
                                                                                                                          output_size=info.pop('output_size'),
                                                                                                                          base_model_output_size=pca_components)
        if checkpoint is not None:
            self.head = MyLightningModuleDelta.load_from_checkpoint(model=self.model, checkpoint_path=checkpoint).model
        else:
            if "output_size" in info:
                info.pop("output_size")
        self.database_name = info.pop('task')
        # Perform Embedding with mini-batch
        mini_batch = 64
        vectors = None
        labels = []

        if from_cache:
            cache_dir = cache_dir + "_" + dataset
            seq1_path, annot_path, label_path = get_cache_delta(dataset, cache_dir)
            y_type = np.float32 if self.database_name == 'regression' else np.int64
            memmap_data = MemMapDatasetDelta(path_seq1=seq1_path,
                                            seq_shape=(info['Seq_length'], pca_components),
                                            chunk_size=info['Seq_length'],
                                            annotation_paths=annot_path,
                                            label_paths=label_path,
                                            label_dtype=y_type)
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
                batch_vectors = batch[0][0]
                if checkpoint is not None:
                    # convert to tensor and move to device
                    batch_vectors = torch.tensor(batch_vectors)
                    batch_vectors = self.head.cache_embed_cnn_delta(batch_vectors, final_pca_components)
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
                if checkpoint is not None:
                    # convert to tensor and move to device
                    vectors_batch = torch.tensor(vectors_batch)
                    vectors_batch = self.head.cache_embed_cnn_delta(vectors_batch, final_pca_components)
                # flatten each vector in the vector batch
                vectors_batch = vectors_batch.reshape(vectors_batch.shape[0], -1)


                if vectors is None:
                    vectors = vectors_batch
                else:
                    vectors = np.vstack((vectors, vectors_batch))
                labels.extend(labels_batch)
        self.vectors = vectors
        if random:
            # randomly the initialization of vectors by creating a vectors with the same shape as vectors but random values
            print(f"Before the Random {self.vectors.shape} {self.vectors.dtype}")
            self.vectors = np.random.rand(*self.vectors.shape).astype(np.float32)
            print(f"After the Random {self.vectors.shape} {self.vectors.dtype}")
        self.labels = labels
        self.init_faiss_index()

    def encode_to_vector(self, data):
        """Returns the difference of the embeddings of the two sequences in the input data.
           and the corresponding labels."""
        res = self.model.cache_embed_delta(data, self.pca_components)
        if self.head is not None:
            res = torch.tensor(res)
            res = self.head.cache_embed_cnn_delta(res, self.pca_components)
        res = res.reshape(res.shape[0], -1)
        return res


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
