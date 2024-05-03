from ..dataloader import data_wrapper as data_wrapper
from ..model_wrapper.base_model import BaseModel
import yaml
import faiss
import numpy as np


class VectorLoader:
    def __init__(self, dataset="clinvar_CLNSIG_hyena-tiny", config_path="/jmain02/home/J2AD015/axf03/zxl79-axf03/repository/genomic-FM-run-exp/genomic-FM/configs/vector_index.yaml"):
        # Load configuration file
        with open(config_path, 'r') as f:
            info = yaml.load(f, Loader=yaml.FullLoader)
        info = info[dataset]
        # Create model
        pca_components = info.pop('pca_components')
        self.model = BaseModel(model_initiator_name=info.pop('model_initiator_name'))
        self.database_name = info.pop('task')
        cls = getattr(data_wrapper, info.pop('class'))
        if 'num_records' in info and 'all_records' in info:
            DATA = cls(num_records=info.pop('num_records'), all_records=info.pop('all_records'))
        else:
            DATA = cls()
        self.data = DATA.get_data(**info)

        # Perform Embedding with mini-batch
        mini_batch = 1000
        vectors = None
        labels = []
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
        return self.model.cache_embed_delta(data, self.model.model.base_model_output_size)


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
