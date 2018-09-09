import numpy as np
import os
import pickle
import pandas as pd
import faiss
faiss.omp_set_num_threads(20)

class Similarity():
    def __init__(self):
        self.xb, self.nb, self.d, self.pt_map = self.load_vectors()
        self.index = faiss.IndexFlatL2(self.d)
        self.index.add(np.ascontiguousarray(self.xb))
            
    def load_vectors(self):
        with open('features.pickle', 'rb') as f:
            data = pickle.load(f)
            xb = data.iloc[:, :-1].values
            xb = xb.astype(np.float32)
            d = xb.shape[1]
            nb = xb.shape[0]
            pt_map = data['0']
            pt_map = pt_map.reset_index()
            pt_map.drop(columns='index', inplace=True)
            return xb, nb, d, pt_map