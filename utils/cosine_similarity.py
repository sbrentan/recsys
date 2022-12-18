import sparse
import numpy as np
import pandas as pd
import re
import sys
import time
from numpy import array
import dask.dataframe as dd
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class CosineSimilarity:

	def __init__(self):
		pass


	def awesome_cossim_top(self, A, B, ntop, lower_bound=0):
		'''
		Performs the cosine similarity between vectors A and B and retuns
		the result as a sparse matrix.
		Args:
			Α           (csr_matrix): The array to be matched (dirty)
			B           (csr_matrix): The baseline array (clean)
			ntop        (int)       : Maximum matches per entry of array A
			lower_bound (int)       : Minimum similarity score to be considered in the returned matrix
		Returns:
			csr_matrix: The matrix (dimensions: len(A)*len(B)) with the similarity scores
		'''
		A = A.tocsr()
		B = B.tocsr()
		# print(A)
		# asdf
		M, _ = A.shape
		_, N = B.shape
		idx_dtype = np.int32
		nnz_max = M * ntop
		indptr = np.zeros(M + 1, dtype=idx_dtype)
		indices = np.zeros(nnz_max, dtype=idx_dtype)
		data = np.zeros(nnz_max, dtype=A.dtype)
		ct.sparse_dot_topn(
			M, N, np.asarray(A.indptr, dtype=idx_dtype),
			np.asarray(A.indices, dtype=idx_dtype),
			A.data,
			np.asarray(B.indptr, dtype=idx_dtype),
			np.asarray(B.indices, dtype=idx_dtype),
			B.data,
			ntop,
			lower_bound,
			indptr, indices, data)
		return csr_matrix((data, indices, indptr), shape=(M, N))

	def get_matches_df(self, sparse_matrix, name_vector, top=100):
		non_zeros = sparse_matrix.nonzero()
		
		sparserows = non_zeros[0]
		sparsecols = non_zeros[1]
		
		if top:
			nr_matches = top
		else:
			nr_matches = sparsecols.size
		
		left_side = np.empty([nr_matches], dtype=object)
		right_side = np.empty([nr_matches], dtype=object)
		similarity = np.zeros(nr_matches)
		
		for index in range(0, nr_matches):
			left_side[index] = name_vector[sparserows[index]]
			right_side[index] = name_vector[sparsecols[index]]
			similarity[index] = sparse_matrix.data[index]
		
		return pd.DataFrame({'left_side': left_side,
							  'right_side': right_side,
							   'similarity': similarity})

	def _example(self, s):

		# print(s)

		# vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
		# tf_idf_matrix_clean = vectorizer.fit_transform(clean)
		# tf_idf_matrix_dirty = vectorizer.transform(dirty)

		# s = dd.map_blocks(sparse.COO)
		# s= dd.from_dask_array(dd.map_blocks(lambda x: csr_matrix(d).todense()))
		# print(s)
		# dtype = s.dtypes[0]
		# s = csr_matrix(s.values)

		if False:
			def ngrams(string, n=3):
				string = re.sub(r'[,-./]|\sBD',r'', string)
				ngrams = zip(*[string[i:] for i in range(n)])
				return [''.join(ngram) for ngram in ngrams]

			import os
			dir_path = os.path.dirname(os.path.realpath(__file__))
			df =  pd.read_csv(dir_path+'/room_types.csv')
			room_types = df['RoomTypes']
			vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
			tf_idf_matrix = vectorizer.fit_transform(room_types)\

			print(tf_idf_matrix[0])
			print(room_types[0])
			print(room_types[29])
			print(room_types[412])	
			asdf

			t1 = time.time()
			print(df['RoomTypes'][0])
			matches = self.awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 10, 0.8)
			t = time.time()-t1
			print("SELFTIMED:", t)

			matches_df = self.get_matches_df(matches, room_types, top=200)
			matches_df = matches_df[matches_df['similarity'] < 0.99999] # For removing all exact matches
			result = matches_df.sort_values(['similarity'], ascending=False).head(10)
			print(result['right_side'])
			print()
			print(result['left_side'])
			print()
			print(result['similarity'])
			asdf


		


		# clusters = np.zeros( (result.shape[0], 11), dtype=np.int8 )
		# clusters = []
		# print(clusters)
		# left_side = list(result['left_side'])
		# right_side = list(result['right_side'])
		# # print(left_side)
		# for i in range(result.shape[0]):
		# 	if(right_side[i] < 0):
		# 		print(right_side[i])
		# 		asdf

		# 	clusters[left_side[i]-1][clusters[left_side[i]-1][0]+1] = right_side[i]
		# 	clusters[left_side[i]-1][0] += 1
		# print(i, clusters[10])


		# print(matches)

		# sys.exit(0)
		# t1 = time.time()
		# doc_term_matrix = s.todense()
		# df = pd.DataFrame(doc_term_matrix, 
		# 	columns=['q'+str(i+1) for i in range(1000)], 
		# 	index=['u'+str(i+1) for i in range(110)])
		# from sklearn.metrics.pairwise import cosine_similarity
		# print(cosine_similarity(df, df))
		# t = time.time()-t1
		# print("SELFTIMED:", t)

		# print(matches)


	def compute(self, s):
		# self._example(s)

		t = time.time()

		# number_of_users = 1110
		# number_of_users = s.shape[0] # for 100 users
		number_of_users = s.shape[0] # for 100 queries

		print(s.shape)
		# asd
		
		tfidf_transformer = TfidfTransformer()
		X = tfidf_transformer.fit_transform(s)
		matches = self.awesome_cossim_top(X, X.transpose(), 10, 0)
		print(matches.shape)
		# matches_df = self.get_matches_df(matches, [i+1 for i in range(number_of_users)], top=10970)
		matches_df = self.get_matches_df(matches, [i+1 for i in range(number_of_users)], top=7132) # for 1000 query_similarity
		# matches_df = self.get_matches_df(matches, [i+1 for i in range(number_of_users)], top=1135)
		# matches_df = self.get_matches_df(matches, [i+1 for i in range(number_of_users)], top=995) # for 100 users
		matches_df = matches_df[matches_df['similarity'] < 0.999]
		# matches_df = matches_df[matches_df['left_side'] != matches_df['right_side']]
		result = matches_df.sort_values(['similarity'], ascending=False)
		print("Time to compute cosine similarity:", round(time.time() - t, 2))

		t = time.time()
		groups = result.groupby('left_side')
		clusters = {}
		empty_clusters = []
		for i in range(number_of_users):
			try:
				clusters[i] = list(groups.get_group(i+1)['right_side'])
			except:
				clusters[i] = []
				empty_clusters.append(i+1)
		print("Time to compute clusters:", round(time.time() - t, 2))

		print(clusters)
		print(empty_clusters)

		# print(clusters)
		return clusters, empty_clusters
