class Glove:
    def __init__(self, vector_size=100):
        self.dims = vector_size
        self.dictionary = self.load_embedding(vector_size)

    def load_embedding(self, vector_size):
        path = "../data/glove/glove.6B." + str(vector_size) + "d.txt"

        dictionary = {}
        with open(path) as f:
            for line in f.readlines():
                key = line[:line.index(" ")]
                value = [float(i) for i in line[line.index(" ") + 1:].split(" ")]

                dictionary[key] = value

        mean_path = "../data/glove/mean_" + str(vector_size) + "d.txt"
        with open(mean_path) as f:
            value = [float(i) for i in f.readline().split(" ")]
            dictionary[""] = value

        return dictionary

    def embed(self, word):
        if word in self.dictionary:
            return self.dictionary[word]
        return self.dictionary[""]
    
    def embed_tokens(self, tokens):
        out = []
        for t in tokens:
            out.append(self.embed(t))
        return out


