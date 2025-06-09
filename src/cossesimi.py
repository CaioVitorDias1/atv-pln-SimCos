
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


reviews = [
    "Produto excelente, chegou rápido e bem embalado. Recomendo!",
    "Não gostei do produto, veio com defeito e o atendimento foi ruim.",
    "Chegou no prazo e funciona bem, mas a embalagem estava rasgada."
]


review1 = [
    "Produto excelente, chegou rápido e bem embalado. Recomendo!"
]

review2 = [
    "Não gostei do produto, veio com defeito e o atendimento foi ruim."
]

review3 = [
    "Chegou no prazo e funciona bem, mas a embalagem estava rasgada."
]

review5estrelas = [
    "Parabéns aos envolvidos, Magazine Luiza pelo site e ofertas, KaBuM! pelo excelente trabalho com a embalagem e MagaLog pela rapidez na entrega. Produto funcionando perfeitamente.",
    "Podem garantir o seu, pois garanti o meu em uma ótima promoção e, sonho de infância realizado.",
    "PS5 excelente! Tem uma diferença considerável na questão de velocidade do processador e imagem (em alguns jogos). Sai do base para o pro. Recomendo para quem quer qualidade."
]

review4estrelas = [
    "Tinha o ps5 base e o salto no pro não é tão grande assim, compensaria se o preço fosse entre 4 a 5 mil, os jogos da Sony estão bem otimizados mas alguns jogos de outras produtoras parecem piores no pro infelizmente, espero que com o tempo fiquem tão bons quanto os da Sony. ",
    "O produto é perfeito rápido e pra quem tem uma tv OLED como eu. A melhora gráfica é muito grande! sim o valor é salgado, Poderia ser mais barato poderia, mas quem gosta e quer da um jeito de pagar. Até no Brasil não existe qualidade sem o valor ser salgado.",
    "So comprei pq sou fan do playstation, mas o valor ainda está muito acima pela falta do leitor e as diferenças da versão base. "

]

review1estrelas =[
    "Sony está de sacanagem lançando o ps5 pro sem leitor de disco e com esse preço, e ainda fala que é o verdadeiro poder do jogos, o ps5 padrão que deveria ser o pro, e não vi muitas mudanças não que vale esse preço.",
    "Não comprem a loja não tá entregando as compras e ainda que receber o valor vem cobrado no cartão de crédito",
    "Game totalmente desnecessário, prometendo o q nao ira entregar, sendo q o ps5 padrão nao usou acho e nem a metade de sua capacidade, ainda nao vi um jogo em nova geração q realmente representasse a nova geração, em fim e isso msm eles bota porque sabe q vender minha opinião"
]

nltk.download('rslp')
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

def preprocess(text):

    text = text.lower()

    text = re.sub(r'[^a-záéíóúàèìòùâêîôûãõç\s]', '', text)

    tokens = text.split()

    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

reviews_cleaned = [preprocess(r) for r in reviews]
for i, r in enumerate(reviews_cleaned):
    print(f"Review {i+1} pré-processado: {r}")



model = SentenceTransformer('all-MiniLM-L6-v2')


embeddings = model.encode(reviews)


def calcular_similaridades(embeddings, texts):
    sim_matrix = cosine_similarity(embeddings)
    n = len(texts)
    for i in range(n):
        for j in range(i+1, n):
            sim = sim_matrix[i][j]
            print(f"Similaridade entre Review {i+1} e Review {j+1}: {sim:.4f}")



sim_matrix = cosine_similarity(embeddings)


df_sim = pd.DataFrame(sim_matrix, index=[f"Review {i+1}" for i in range(len(reviews))],
                                   columns=[f"Review {i+1}" for i in range(len(reviews))])





calcular_similaridades(embeddings, reviews)

print("Matriz de Similaridade entre Reviews (valores entre 0 e 1):")
print(df_sim)