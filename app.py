import gensim
from flask import Flask, render_template, request
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import brown

model_path = 'model_gl.bin'

# Specify binary=True only if the model is in binary format
try:
    glove_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
except UnicodeDecodeError:
    pass

my_file = open("tbbt_wiki.txt", "r") 
  
# reading the file containg information
data = my_file.read()
#convert data into list
data_into_list = data.replace('\n', ' ').split(".") 

def tokenize_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize the words using nltk's word_tokenize function
    words = word_tokenize(text)

    return words


file_path = 'tbbt_wiki.txt'
tokenized_words = tokenize_text(file_path)
tbbt_vocab = set(tokenized_words)

nltk.download('brown')
corpus = brown.sents(categories=['news'])
flatten = lambda l: [item for sublist in l for item in sublist]
vocabs = list(set(flatten(corpus))) 
word2index = {v:idx for idx, v in enumerate(vocabs)}
app = Flask(__name__)



def get_embed(model, word):
    id_tensor = torch.LongTensor([word2index[word]])
    v_embed = model.center_embedding(id_tensor)
    u_embed = model.outside_embedding(id_tensor)
    word_embed = (v_embed + u_embed) / 2
    x, y = word_embed[0][0].item(), word_embed[0][1].item()
    return x, y

def cos_sim(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def search_similar(input):
    word_dict={}
    try:
        #use glove model
        input_embed = get_embed(glove_model, input)
        for word in tbbt_vocab:
            if word in vocabs:
                word_dict[word] =  cos_sim(input_embed, get_embed(glove_model, word))
            else:
                continue

        sorted_dict = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True))
    except:
        print('There is no word in the dictionary, please type new word')

    return list(sorted_dict)[:10]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            search_query = request.form['search']
        
            result = f"Search Result: {search_similar(search_query)}"
            return render_template('index.html', result=result)
        except :
            pass


    # return render_template('index.html', result=None)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
