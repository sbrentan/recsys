import os
import sys
import itertools
import numpy as np
import pandas as pd
import time

import random
from math import sqrt

from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD

from utils.io_manager import IOManager
from utils.generator import MovieGenerator
from utils.data_manager import DataManager
from utils.optimization import Optimization
from utils.cosine_similarity import CosineSimilarity

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# pyarrow -----
# unidecode ------

# asdf
class Recommendator:

    _io = None
    _data = None

    TEST = 'test_3'

    CONVERT_MOVIES   = False
    GENERATE_QUERIES = False
    GENERATE_VOTES  = False

    SKIP_READINGS = True

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._io = IOManager(dir_path + "/datasets/")
        self._data = DataManager(self._io)

        t = time.time()
        self._generator = generator = MovieGenerator(self._io, self._data)
        if(self.CONVERT_MOVIES or self.GENERATE_QUERIES or self.GENERATE_VOTES):
            if(self.CONVERT_MOVIES):   generator.convert_movies(source="movies_metadata.csv", dest="films.csv")
            if(self.GENERATE_QUERIES): generator.generate_queries(dest="queries.csv")
            if(not self.SKIP_READINGS): self._data.read_inputs()
            if(self.GENERATE_VOTES):  generator.generate_utilmat(size=10000, dest="utilmat5.csv")
        else:
            if(not self.SKIP_READINGS): self._data.read_inputs()

        # self._data.read_utilmat()
        print("Time to read: " + str(round(time.time() - t, 2)))
        # sys.exit(0)


    def _hierarchical_clustering(self, utilmat, mat_means, threshold = 0, nclusters = None):
        # utilmat = np.floor(utilmat / mat_means).round(0).astype(np.int8)

        # REQUIRES UTILMAT ALREADY .fillna(0)
        utilmat = utilmat.astype(bool).astype(np.int8)

        final_clusters = {}
        qclusters = [{ 'cert': "", 'mat': utilmat, 'qids': [] }]
        while(len(qclusters) > 0):
            # clust = qclusters.pop(0)
            cdict = [len(c['mat']) for c in qclusters]
            clustid = np.argmax(cdict)
            clust = qclusters[clustid]
            del qclusters[clustid]
            mat = clust['mat']
            # print(len(mat))
            if(len(mat) > threshold and (nclusters == None or len(final_clusters) + len(qclusters) <= nclusters)):
                maxes = mat.sum(axis=0)
                idmax = maxes.idxmax()
                while(self._data.queries_ids[idmax] in clust['qids']):
                    maxes[idmax] = 0
                    idmax = maxes.idxmax()
                idmax = self._data.queries_ids[idmax]

                cond = (mat.iloc[:, idmax] == 0)
                qclusters.append({ 'cert': clust['cert']+'0', 'mat': mat[cond], 'qids': clust['qids'] + [idmax]})

                cond = (cond - 1).astype(bool)
                qclusters.append({ 'cert': clust['cert']+'1', 'mat': mat[cond], 'qids': clust['qids'] + [idmax]})
            else:
                final_clusters[clust['cert']] = mat.index

        check = { x: len(mat) for x, mat in final_clusters.items() }
        print(sum([x for c, x in check.items()]), len(check))
        print(check)
        return final_clusters


    def _compute_users_clusters(self, mat):
        # t = time.time()
        # cols, mat = self._io.input_csr_matrix(self.TEST+'/utilmat.csv')
        # print("Time to read csr matrix: "+str(round(time.time() - t, 3)))
        #100 -> 10000, 0.016 -> 1.6. 16s to read 100k
        # mat = self._data.

        t = time.time()
        cossim = CosineSimilarity()
        clusters, empty_clusters = cossim.compute(mat)

        mat = mat.todense()
        # print("Time to cossim and todense: "+str(round(time.time() - t, 3)))

        t = time.time()
        results, similarities = [], []
        for c, values in clusters.items():
            # calculate average votes for the clustered users
            a = np.empty((len(clusters[0]), mat.shape[1]))
            counter = np.zeros((1, mat.shape[1]))
            avg_sim, sim_count = 0, 0
            for ind, (userid, sim) in enumerate(values.items()):
                b = mat[userid-1]
                counter += b.astype(bool)
                a[ind] = b
                # avg_sim += sim
                avg_sim = max(avg_sim, sim)
                sim_count += 1
            avg_sim = round(avg_sim, 2)
            # avg_sim = round(avg_sim/sim_count, 2) if sim_count != 0 else 0

            summed_values = a.sum(axis=0)
            result  = (summed_values / counter).round(2)
            results.append(result)
            similarities.append(avg_sim)
            # print(counter)
            # print(summed_values)
            # print(result)
            # asdf
            # sys.exit(0)
            # counter = [len(a[:, x].nonzero()[0]) for x in range(mat.shape[1])]

            # a = lil_matrix((len(clusters[0]), mat.shape[1]), dtype=np.int8)
            # np.set_printoptions(threshold=sys.maxsize)
            # col_values = a[:, 0]
            # print(counter)
            # asdf
            # result = round(sum(col_values) / len(col_values.nonzero()[0]), 2)
        # print("Time to compute utilmat values: " + str(round(time.time() - t, 3)))
        return results, similarities

    def _compute_queries_clusters(self):
        movies_indexes = self._data.movies_df.index
        self._data.movies_df.index = [x for x in range(len(movies_indexes))]


        # print(self._data.movies_df)
        # asdf

        t = time.time()
        cols = self._data.movies_df.columns
        shingles = []
        c = 0
        for q in self._data.queries:
            # panda = self._data.movies_df
            # for cond in q[1::]:
            #     col, val = cond.split("=")
            #     panda = panda[panda[col] == val]
            #     # l.append(col+" == '"+val+"'")
            panda = self._data.get_as_from_panda(query=q)
            new = np.zeros(self._data.movies_df.shape[0])
            if(not panda.empty):
                indexed = panda.index
                # print(self._data.queries_idsself._queries[i][0])
                for e in indexed:
                    # print(panda.index)
                    # new[self._data.movies_ids[e]] = 1
                    new[e] = 1
            else:
                c += 1
            shingles.append(new)

        mat = csr_matrix(shingles, dtype=np.int32)
        print('Time to compute shingles: '+str(time.time() - t))



        # mat = pd.DataFrame(shingles)

        t = time.time()
        svd = TruncatedSVD(n_components=100)
        svd.fit(mat)
        mat = svd.transform(mat)
        print('time to svd: '+str(time.time()-t))

        # sparse_utilmat = utilmat.astype(pd.SparseDtype("float64",0)).sparse.to_coo().tocsr()
        # cossim = CosineSimilarity()
        # clusters, empty = cossim.compute(sparse_utilmat)

        t = time.time()
        cossim = CosineSimilarity()
        clusters, empty = cossim.compute(mat)
        print('time to cossimmm: '+str(time.time()-t))

        return clusters, empty
        # time.sleep(10)


    def _hybrid_standardization(self):
        pass

    def _hybrid_collaborative(self, mat_means):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        # opt = Optimization(dir_path, self.TEST)
        # lclusters = opt.optimize_mat(mat = self._data.utilmat_df, ncertificates = 10, nsplits = 10)
        
        threshold = int(len(self._data.utilmat_df) / 10)
        nclusters = int(sqrt(len(self._data.utilmat_df)))
        print(threshold, nclusters)
        lclusters = self._hierarchical_clustering(self._data.utilmat_df, mat_means, threshold = threshold, nclusters = nclusters)

        votes = np.zeros(self._data.utilmat_df.shape)
        count = np.zeros(self._data.utilmat_df.shape)

        sparse_utilmat = self._data.utilmat_df.astype(pd.SparseDtype("float64",0)).sparse.to_coo().tocsr()
        for cert, users in lclusters.items():
            ids = [self._data.user_ids[uid] for uid in users]
            mat = sparse_utilmat[ids]

            results, similarities = self._compute_users_clusters(mat)
            for i in range(len(results)):
                votes[ids[i]] += results[i][0]
                count[ids[i]] += 1
        votes = np.around(votes/count, 0)
        # print(votes)
        return pd.DataFrame(votes, index=self._data.user_ids, columns=self._data.queries_ids)



        # queries_clusters = self._compute_queries_clusters()
        # 3.5s for 10k users
        # users_clusters = self._compute_users_clusters()
        # print(users_clusters)

    def _hybrid_content_based(self, bias, mm):

        # mat_means = mm.copy()

        clusters, empty = self._compute_queries_clusters()

        df_bool = self._data.utilmat_df.astype(bool).astype(int)
        
        t = time.time()
        # counter = pd.DataFrame(np.ones(self._data.utilmat_df_T.shape), index=self._data.queries_ids, columns=self._data.user_ids)
        mat_means = pd.DataFrame(np.zeros(self._data.utilmat_df.shape), index=self._data.user_ids, columns=self._data.queries_ids)
        for c, values in clusters.items():
            queryids = map(lambda x: x-1, list(values.keys()))
            mean = bias.iloc[:, queryids].mean(axis=1, skipna=True)
            mat_means.iloc[:, c] = mean + mm.iloc[:, c]


        print('Time to compute values: '+str(time.time()-t))
        # mat_means = (mat_means / counter).round(2)

        mat_means = mat_means.fillna(mm)

        print(mat_means)

        return mat_means



    def _hybrid(self):
        # ==================       READING       ================== #
        t = time.time()
        self._data.read_test_inputs(self.TEST)
        print('Time for reading:', round(time.time() - t, 2))

        # ==================   STANDARDIZATION   ================== #
        q_means = self._data.utilmat_df.mean(axis = 0, skipna = True)
        u_means = self._data.utilmat_df.mean(axis = 1, skipna = True)

        # self._data._utilmat_df_T = self._data.utilmat_df_T.fillna(0)
        shape = (len(self._data.user_ids), len(self._data.queries))
        arr = np.zeros(shape)
        # arr = np.zeros(self._data.utilmat_df_T.shape)
        q_means_list = q_means.tolist()
        for i in range(len(arr)):
            arr[i] = q_means_list
            # arr[i] = q_means_list[i]
        mat_means = pd.DataFrame(arr, index=self._data.user_ids, columns=self._data.queries_ids)
        # mat_means = pd.DataFrame(arr, index=self._data.queries_ids, columns=self._data.user_ids)

        # bias = (self._data.utilmat_df - mat_means) * self._data.utilmat_df.astype(bool).astype(int)
        bias = self._data.utilmat_df - mat_means

        # bias = (self._data.utilmat_df_T - mat_means) * self._data.utilmat_df_T.astype(bool).astype(int)


        self._data._utilmat_df = self._data.utilmat_df.fillna(0)


        # ==================    COLLABORATIVE    ================== #
        # For users
        t = time.time()
        mat1 = self._hybrid_collaborative(mat_means).fillna(0)
        print('Time to compute collaborative:', round(time.time() - t, 2))



        # ==================    CONTENT-BASED    ================== #
        # For query-films shingles
        t = time.time()
        mat2 = self._hybrid_content_based(bias, mat_means)
        print('Time to compute content-based:', round(time.time() - t, 2))
        # print(mat2)


        collaborative_ratio = 0.1
        mat1_bool = mat1.astype(bool).astype(int)
        mat1_ones = pd.DataFrame(np.ones(mat1_bool.shape), index=self._data.user_ids, columns=self._data.queries_ids)
        final_mat = (mat1 * collaborative_ratio + mat2 * (mat1_ones - (mat1_bool * collaborative_ratio))).round(0)

        print((mat1_bool * collaborative_ratio))
        print((mat1_ones - (mat1_bool * collaborative_ratio)))

        print('mat1')
        print(mat1)

        print('mat2')
        print(mat2)


        print('final_mat')
        print(final_mat)

        # asdf
        return final_mat





    def recommend(self):

        # self._using_users_similarity()

        # self._using_query_similarity()

        # self._using_mat_avgs()

        # self._using_lsh()

        return self._hybrid()


        # CONTENT BASED
        # create query-query affinity matrix
        # for every query:
        #       assign to the reachable items the query id
        # for every item:
        #       for every pair of reachable queries: add +1 in the affinity matrix
        # compute transitive affinity alongside user feedback
        # for every feedbacked query:
        #       get most affine query
        #       if feedback not present in utilmat:
        #          assign score to that query in utilmat by a factor of (score/100)
        #           --->>>> PROBLEM: does not account for single film score
        #                --->>> SOLUTION: change query score to average from film score (assigned from user feedback) 
        # self._using_affinity_matrix()



        # HYBRID BASED
        # cluster similar users together
        #       1) TFIDF based on film feedbacks
        #       2) Cluster film together and observe users feedbacks on those clusters
        #           * Films can be grouped together based on similar attributes? and names?
        # compute content based on representative user for every cluster



        # QUERY CORRESPONDANCE
        # find queries correspondance from each other
        # create query-query correspondance matrix:
        # for every user:
        #       for every feedback:
        #               find other feedback
        #               compute correspondance similarity between said queries
        # for every user:
        #       for every evaluated query:
        #               for every other query:
        #                       compute evaluation based on qq_correspondance matrix
        #               find average of those evaluations
        # self._using_correspondance_matrix()





        # save similarity of pair of items inside sparse matrix
        # for every answer set:
        #       raise/decrease similarity of each pair of items by an amount related to feedback (e.g. + (feedback-50) / 50)
        # 
        # compute matrix from item similars and user feedback
        # for every user answer set:
        #       give score to most connected items up to a maximum of user feedback
        # fetch items with highest score
        # compute query that better represents those items

        print()
        pass


    def evaluate(self, mat):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        print(mat)
        if(mat):
            real_votes = pd.read_csv(dir_path + '/datasets/' + self.TEST+'/real_votes.csv', header=None)
            real_votes.index = ['u'+str(i+1) for i in range(real_votes.shape[0])]
            real_votes.columns = ['q'+str(i+1) for i in range(real_votes.shape[1])]

            print('real_votes')
            print(real_votes)
            print((mat - real_votes).abs())
            print((mat - real_votes).abs().stack().mean())
        else:

            f = open(dir_path + '/datasets/' + self.TEST+'/definitions.csv', "r", encoding='utf-8')
            line = f.readline()
            definitions = []
            while(line):
                definitions.append(eval(line))
                line = f.readline()
            f.close()
            self._data.read_test_inputs(self.TEST)
            self._generator._efficient_users_queries_votes(definitions, self._data.queries, self._data.movies_df, dir_path, self.TEST)
            asdf
            print('Finished reading inputs')
            real_votes = np.zeros((len(definitions), len(self._data.queries)))
            print(real_votes.shape)
            limit = 999
            for row in range(len(real_votes)):
                print(row, len(real_votes))
                for col in range(len(real_votes[row])):
                    # if(mat.iloc[row, col] != np.nan):
                    # print(self._data.queries[col])
                    real_vote = self._generator._simulate_query_vote(definitions[row], self._data.queries[col], True)
                    real_votes[row][col] = real_vote
                if(limit != None and row == limit):
                    break
            f = open(dir_path + '/datasets/' + self.TEST+'/real_votes.csv', "w")
            for ind, row in enumerate(real_votes):
                f.write(", ".join(map(lambda x: str(x), row)) + '\n')
                if(limit != None and ind == limit):
                    break
            f.close()


        

if __name__ == "__main__":

    # t = time.time()
    # from utils.optimization import Optimization

    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # opt = Optimization(dir_path)

    # print()
    # print(time.time() - t)
    # sys.exit(0)


    t = time.time()
    recommendator = Recommendator()
    # mat = recommendator.recommend()
    print(time.time() - t)

    print()
    recommendator.evaluate(None)


# WHAT HAPPENS WHEN N. QUERIES < N. UTILMAT