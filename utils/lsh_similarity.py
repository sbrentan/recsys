import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSHForest

class LSHSimilarity:


	def __init__():
		db = pd.read_csv('papers.csv')
		db['text'] = db['title'] + ' ' + db['abstract']
		forest = get_forest(db, permutations)
		pass

	def get_forest(data, perms):
		start_time = time.time()
		
		minhash = []
		
		m = MinHash(num_perm=perms)
		for aset in data:
			# tokens = preprocess(text)
			for fid in aset:
				m.update(fid.encode('utf8'))
			minhash.append(m)

		print(aset)
			
		forest = MinHashLSHForest(num_perm=perms)
		
		for i,m in enumerate(minhash):
			forest.add(i,m)
			
		forest.index()
		
		print('It took %s seconds to build forest.' %(time.time()-start_time))
		
		return forest

	def predict(aset, database, perms, num_results, forest):
		start_time = time.time()
		
		# tokens = preprocess(text)
		m = MinHash(num_perm=perms)
		for fid in aset:
			m.update(fid.encode('utf8'))
			
		print(m)
		idx_array = np.array(forest.query(m, num_results))
		if len(idx_array) == 0:
			return None # if your query is empty, return none
		
		# result = database.iloc[idx_array]['title']
		result = idx_array
		
		print('It took %s seconds to query forest.' %(time.time()-start_time))
		
		return result