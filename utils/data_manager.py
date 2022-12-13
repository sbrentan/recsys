class DataManager():

	_io = None

	_users = {}
	_queries = {}
	_films = {}
	_utilmat = {}

	def read_inputs(self):
		for l in ["users", "queries", "films"]:
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


	def get_answer_set(self, query_id):
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
	def films(self):
		return self._films
	
	@property
	def utilmat(self):
		return self._utilmat
	
	
	