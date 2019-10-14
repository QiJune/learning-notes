class KVStore(object):
    def __init__(self):
        self.param_db = {}
        self.embedding_param_db = {}

    def get_param(self, name):
        return self.param_db[name]

    def set_param(self, name, value):
        self.param_db[name] = value

    def get_embedding_param(self, name, ids):
        param = self.embedding_param_db[name].get(ids)
        return param

    def set_embedding_param(self, name, ids, value):
        self.embedding_param_db[name].set(ids, value)

    def set_embedding_table(self, name, table):
        self.embedding_param_db[name] = table
