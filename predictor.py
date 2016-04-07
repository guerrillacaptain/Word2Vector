import gensim,logging,numpy,sys,math
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
reload(sys)
sys.setdefaultencoding('utf-8')

def init_dict(file):
    dict = {}
    for line in open(file):
        temp = line.strip().split('|')
        dict[temp[0]] = "|".join(temp[2:])
    return dict

def cos(vec_1,vec_2):
    np_vec_1 = numpy.array(vec_1)
    np_vec_2 = numpy.array(vec_2)
    num = float(numpy.dot(np_vec_1,np_vec_2))
    demon = numpy.linalg.norm(np_vec_1) * numpy.linalg.norm(np_vec_2)
    cos = num / demon
    sim = 0.5 +0.5 *cos
    return "%.5f" % sim

##D(vec_2||vec_1)
def kl_divergence(vec_1,vec_2):
    return "%.10f" % sum(vec_2[x] * math.log((vec_2[x]) / (vec_1[x])) for x in range(len(vec_2)) if vec_2[x] != 0.0 and vec_1[x] != 0)

def predict(source,dest,model,dict):
    output = open(dest,'w')
    for line in open(source):
        temp = line.strip().split('|')
        if dict.has_key(temp[1]) and dict.has_key(temp[2]):
            sentence_1 = dict[temp[1]]
            sentence_2 = dict[temp[2]]
            vector_1 = numpy.zeros(32)
            vector_2 = numpy.zeros(32)
            num_1 = 0
            num_2 = 0
            for word in sentence_1.split(" "):
                if len(word.strip()) > 0:
                    try:
                        vector_1 = vector_1 + model[word]
                        num_1 +=1
                    except KeyError:
                        continue
            for word in sentence_2.split(" "):
                if len(word.strip()) > 0:
                    try:
                        vector_2 = vector_2 + model[word]
                        num_2 +=1
                    except KeyError:
                        continue
            vector_1 = vector_1/num_1
            vector_2 = vector_2/num_2
            similarity = cos(vector_1,vector_2)
            #similarity = kl_divergence(vector_1,vector_2)
            output.write(similarity+" "+line)
        else:
            continue

if __name__ == '__main__':
    modelname="item.word2vec"
    model = gensim.models.Word2Vec.load(modelname)
    #dict = init_dict("/export/App/yaojiandong/evaluate/cut_document.dict")
    dict = init_dict("/export/App/yaojiandong/evaluate/cut_document_0728.dict")
    predict(sys.argv[1],sys.argv[2],model,dict)

