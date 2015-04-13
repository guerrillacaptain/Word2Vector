import gensim,logging,jieba
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
    
    def __iter__(self):
        for line in open(self.filename):
            yield list(jieba.cut(line.strip()))



def train_model(filename,modelname):
    sentences = MySentences(filename)
    model = gensim.models.Word2Vec(sentences)
    model.save(modelname)

def load_model(modelname):
    model = gensim.models.Word2Vec.load(modelname)



if __name__ == '__main__':
    filename="sku_name.txt"
    modelname="item_name.word2vec"
    train_model(filename,modelname)
    #load_model(modelname)
