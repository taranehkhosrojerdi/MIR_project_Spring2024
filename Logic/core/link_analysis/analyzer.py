from Logic.core.link_analysis.graph import LinkGraph
from Logic.core.indexer.indexes_enum import Indexes
from Logic.core.indexer.index_reader import Index_reader

class LinkAnalyzer:
    def __init__(self, root_set):
        self.root_set = {movie['id']: movie for movie in root_set}
        self.graph = LinkGraph()
        self.hubs = set()
        self.authorities = set()
        self.initiate_params(root_set)

    def initiate_params(self, root_set):
        for movie in root_set:
            self.graph.add_node(movie['id'])
            self.hubs.add(movie['id'])
            for star in movie['stars']:
                self.graph.add_node(star)
                self.graph.add_edge(movie['id'], star)
                self.authorities.add(star)

    def expand_graph(self, corpus):
        existing_movies = set(self.root_set.keys())
        starred_by_roots = {hub: set(self.graph.get_successors(hub)) for hub in self.hubs}

        for movie in corpus:
            if movie['id'] not in existing_movies:
                common_stars = any(star in starred_by_roots[hub] for hub in self.hubs for star in movie['stars'])
                if common_stars:
                    self.graph.add_node(movie['id'])
                    self.root_set[movie['id']] = movie
                    self.hubs.add(movie['id'])
                    for star in movie['stars']:
                        if star not in self.authorities:
                            self.graph.add_node(star)
                            self.authorities.add(star)
                        self.graph.add_edge(movie['id'], star)
                    starred_by_roots[movie['id']] = set(movie['stars'])

    def hits(self, num_iteration=5, max_result=10):
        a_s = {node: 1.0 for node in self.authorities}
        h_s = {node: 1.0 for node in self.hubs}

        for _ in range(num_iteration):
            h_s = {hub: sum(a_s[auth] for auth in self.graph.get_successors(hub)) for hub in self.hubs}
            a_s = {auth: sum(h_s[hub] for hub in self.graph.get_predecessors(auth)) for auth in self.authorities}

        sorted_a_s = sorted(a_s.items(), key=lambda x: x[1], reverse=True)[:max_result]
        sorted_h_s = sorted(h_s.items(), key=lambda x: x[1], reverse=True)[:max_result]

        top_authorities = [auth for auth, score in sorted_a_s]
        top_hubs = [hub for hub, score in sorted_h_s]

        return top_authorities, top_hubs

# if __name__ == "__main__":
#     file_path = 'Logic/core/indexer/index/documents_index.json'
#     with open(file_path, 'r') as file:
#         corpus = list(json.load(file).values())[:10]
#     root_set = corpus[:2]

#     analyzer = LinkAnalyzer(root_set=root_set)
#     analyzer.expand_graph(corpus=corpus)
#     actors, movies = analyzer.hits(max_result=5)

#     print("Top Actors:")
#     print(*actors, sep=' - ')
#     print("Top Movies:")
#     print(*movies, sep=' - ')
