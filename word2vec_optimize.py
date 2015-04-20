import gensim,logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, files):
        self.files = files
    
    def __iter__(self):
        for file in self.files:
            for line in open(file):
                line_list = line.split("|")
                pending_doc = "|".join(line_list[2:])
                if len(pending_doc) > 0:
                    yield pending_doc.split(" ")
                else:
                    continue



def train_model(files,modelname):
    sentences = MySentences(files)
    model = gensim.models.Word2Vec(sentences,size=500,min_count=10,workers=10)
    model.save(modelname)

def load_model(modelname):
    model = gensim.models.Word2Vec.load(modelname)



if __name__ == '__main__':
    files=["item.txt"]
    modelname="item.word2vec"
    train_model(files,modelname)
    #load_model(modelname)
