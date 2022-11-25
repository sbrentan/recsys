import os
import sys
import itertools

from utils.io_manager import IOManager
from utils.data_manager import DataManager

class Recommendator:

    _io = None
    _data = None

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._io = IOManager(dir_path + "/datasets/")
        self._data = DataManager(self._io)

    def _using_affinity_matrix(self):
        for_user = list(self._data.utilmat['users'].items())[0][0]
        aff_mat = {}
        for qid, q in self._data.queries.items():
            items = self._data.get_answer_set(qid)
            q["as_ids"] = items
            for fid in items:
                i = self._data.films[fid]
                if("queries" in i): i["queries"].add(qid)
                else: i["queries"] = set([qid])
                if('score' not in i): i['score'] = 0
                user_fb = int(self._data.utilmat['users'][for_user][self._data.utilmat['queries'][qid]])
                i['score'] = max(user_fb, i['score'])
        for fid, i in self._data.films.items():
            comb = itertools.combinations(i["queries"],2)
            for c in comb:
                first, second = (c[1], c[0]) if c[0] > c[1] else (c[0], c[1])
                if(first not in aff_mat): aff_mat[first] = {}
                if(second not in aff_mat[first]):
                    aff_mat[first][second] = 1
                else:
                    aff_mat[first][second] += 1
        for uid, fblist in self._data.utilmat['users'].items():
            for qid, qindex in self._data.utilmat['queries'].items():
                for qid, q in self._data.queries.items():
                    if(q['as_ids']):
                        q['score'] = float(sum(list(map(lambda x: self._data.films[x]['score'], q['as_ids'])))) / len(q['as_ids'])
                    else:
                        q['score'] = 0
        print([(qid, q['score']) for qid, q in self._data.queries.items()])



    def recommend(self):

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
        self._using_affinity_matrix()



        # HYBRID BASED
        # cluster similar users together
        #       1) TFIDF based on film feedbacks
        #       2) Cluster film together and observe users feedbacks on those clusters
        #           * Films can be grouped together based on similar attributes? and names?
        # compute content based on representative user for every cluster



        # save similarity of pair of items inside sparse matrix
        # for every answer set:
        #       raise/decrease similarity of each pair of items by an amount related to feedback (e.g. + (feedback-50) / 50)
        # 
        # compute matrix from item similars and user feedback
        # for every user answer set:
        #       give score to most connected items up to a maximum of user feedback
        # fetch items with highest score
        # compute query that better represents those items

        pass

        

if __name__ == "__main__":
    recommendator = Recommendator()
    recommendator.recommend()


# WHAT HAPPENS WHEN N. QUERIES < N. UTILMAT