import numpy as np
import pandas as pd

class DataManager():

	_io = None

	_users = None
	_queries = None
	_films = None
	_utilmat = None

	_movies = None
	_mclusters = None
	_queries_ids = None

	_utilmat_df = None
	_movies_df = None
	_movies_ids = None

	def read_inputs_old(self):
		self._queries = {}
		self._users = {}
		self._films = {}
		self._utilmat = {}
		# for l in ["users", "queries", "films"]:
		for l in ["users", "queries"]:
			columns, items = self._io.input(l+".csv", True if l == "films" else False)
			for i in items:
				if(not columns):
					getattr(self, "_"+l)[i[0]] = {'val' : i[1::]}
				else:
					getattr(self, "_"+l)[i[0]] = {}
					for ind, c in enumerate(columns):
						if(c == 'genres' or c == 'companies'):
							getattr(self, "_"+l)[i[0]][c] = i[ind+1].split("~")
						else:
							getattr(self, "_"+l)[i[0]][c] = i[ind+1]


	def __init__(self, io_manager):
		self._io = io_manager
		# self.read_inputs()

		# print(self._users)
		# print(self._films)
		# print(self._queries)
		# print(self._utilmat)


	def read_utilmat(self):
		_, mat = self._io.input("utilmat.csv")
		self._utilmat['queries'] = {}
		self._utilmat['users'] = {}
		for ind, q in enumerate(mat[0]):
			self._utilmat['queries'][q] = ind
		for i in range(len(mat[1::])):
			self._utilmat['users'][mat[i+1][0]] = [int(x) if x else -1 for x in mat[i+1][1::]]


	def read_inputs(self):
		self._movies, self._mclusters, columns, features, fcounter = self._io.input_movies("films.csv")
		column_ids = {x: ind+1 for ind, x in enumerate(columns)}
		# print({i for i in features[5] if features[5][i]==75})
		# print(features[5]['75'])
		# print(columns)
		# asdf
		# print(column_ids)
		# asdf
		# print(columns)
		_, queries = self._io.input("queries.csv")
		# print(queries[0])
		# print(fcounter)
		self._queries = []
		self._queries_ids = {}
		counter = 0
		for conditions in queries:
			as_int_set = np.zeros(len(columns), dtype=np.int32)
			for c in conditions[1::]:
				col, val = c.split('=')
				col_id = column_ids[col]
				if(val not in features[col_id]):
					features[col_id][val] = fcounter[col_id-1]
					as_int_set[col_id-1] = fcounter[col_id-1]
					fcounter[col_id-1] += 1
				else:
					as_int_set[col_id-1] = features[col_id][val]

			# print(as_int_set)
			self._queries.append(as_int_set)
			self._queries_ids[conditions[0]] = counter
			counter += 1
		# print(len(self._queries))
		# print(self._queries) 


	def read_pd_inputs(self):
		_, self._queries = self._io.input("queries.csv")
		self._queries_ids = {}
		for i in range(len(self._queries)):
			self._queries_ids[self._queries[i][0]] = i

		# self._utilmat_df = pd.read_csv(self._io.dataset_path + 'utilmat.csv').astype(np.str)
		self._utilmat_df = pd.read_csv(self._io.dataset_path + 'utilmat5.csv')
		# self._movies_df = pd.read_csv(self._io.dataset_path + 'films.csv').astype(np.str)
		self._movies_df = pd.read_csv(self._io.dataset_path + 'films.csv')
		self._movies_ids = {}
		indexes = self._movies_df.index
		for i in range(len(indexes)):
			self._movies_ids[indexes[i]] = i



	def get_as_from_panda(self, query_id=0, query=None):
		if(query_id != 0):
			# print(asdf)
			query = self._queries[query_id]
		panda = self._movies_df
		cols = self._movies_df.columns
		is_int = True
		for cond in query[1::]:
			col, val = cond.split("=")
			if(is_int):
				try: val = int(val)
				except: is_int = False
			# print(col, val)
			panda = panda[panda[col] == val]
		return panda


	def get_answer_set(self, query_id=0, query=None):
		if(query_id == 0):
			query = self._queries[query_id]
		# position of condition to analyze
		cond_ind = 0
		# useless
		prev_clusters = [None for x in range(len(query))]
		# every list contains remaining elements to check. every loop the first is
		cluster_values = [[] for x in range(len(query))]
		aset = set()
		cluster = self._mclusters
		# query = [0, 0, 0, 0, 80]
		cluster_values[0] = list(cluster.keys()) if query[0] == 0 else [query[0]]
		# print(cluster_values)
		# print(query)
		# asdf
		# aset.update([1000])
		while(True):
			# print(cond_ind, [cluster_values[i][0] if cluster_values[i] else 0 for i in range(len(query))])
			if(cond_ind == len(query)):
				aset.update(cluster)
				cond_ind -= 1
			elif(cond_ind == -1):
				break
			else:
				# if(query[cond_ind] == 0):
				if(cluster_values[cond_ind]):
					# print('check:',cluster_values[cond_ind][0], cluster)
					prev_clusters[cond_ind] = cluster
					cluster = cluster[cluster_values[cond_ind][0]]
					cluster_values[cond_ind].pop(0)
					cond_ind += 1
					if(cond_ind < len(query)):
						if(query[cond_ind] == 0):
							cluster_values[cond_ind] = list(cluster.keys())
						else:
							# print('check2:', cond_ind, cluster)
							if(query[cond_ind] in cluster):
								cluster_values[cond_ind] = [query[cond_ind]]
				else:
					cond_ind -= 1
					if(cond_ind >= 0):
						cluster = prev_clusters[cond_ind]
				# else:
				# 	if(cluster_values)
			
			# print('----------')
			# print('cluster_values:', cluster_values)
			# print('----------')
			# print('cond_ind:', cond_ind)
			# print('----------')
			# print('cluster:', cluster)
			# print('---------------------------')
			# asdf
		# print(aset)
		return aset
		

	def get_as_from_movies(self, query_id=0, query=None):
		if(query_id == 0):
			query = self._queries[query_id]
		# query = [0, 0, 0, 0, 80]
		result = []
		# print(self._movies)
		for mid, m in self._movies.items():
			skip = False
			# if(mid == 7106):
			# 	print(mid, m)
			for i in range(len(query)):
				if(query[i] != 0 and m[i+1] != query[i]):
					skip = True
					break
			if(not skip):
				result.append(mid)
		return result
		# print(result)



	def get_answer_set_old(self, query_id):
		result = []
		conditions = []
		# print(self._queries[query_id])
		# for cond in self._queries[query_id]['val'][0].split(","):
		for cond in self._queries[query_id]['val']:
			field, value = cond.split("=")
			conditions.append((field, value))

		for fid, f in self._films.items():
			skip = False
			for cfield, cvalue in conditions:
				# print(cvalue, f['genres'])
				# print(cvalue, f['companies'])
				if(cfield == "genres"):
					skip = cvalue not in f["genres"]
				elif(cfield == "companies"):
					skip = cvalue not in f["companies"]
				elif(f[cfield] != cvalue):
					skip = True
				if(skip): break
			if (not skip):
				result.append(fid)
			# asdf
		return result


	@property
	def users(self):
		return self._users

	@property
	def queries(self):
		return self._queries

	@property
	def queries_ids(self):
		return self._queries_ids
	
	
	@property
	def films(self):
		return self._films
	
	@property
	def utilmat(self):
		return self._utilmat

	@property
	def movies(self):
		return self._movies
	
	@property
	def mclusters(self):
		return self._mclusters
	
	
	
	@property
	def utilmat_df(self):
		return self._utilmat_df
	

	@property
	def movies_df(self):
		return self._movies_df

	@property
	def movies_ids(self):
		return self._movies_ids
	
	