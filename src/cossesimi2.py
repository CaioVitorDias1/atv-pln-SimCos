

import re
import nltk
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sentence_transformers import SentenceTransformer


nltk.download('stopwords')
nltk.download('rslp')


stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


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


reviews_1_proc = [preprocess(r) for r in review1estrelas]
reviews_4_proc = [preprocess(r) for r in review4estrelas]
reviews_5_proc = [preprocess(r) for r in review5estrelas]


for i, r in enumerate(reviews_1_proc, 1):
    print(f"Review de {i} estrelas pré-processado:", r)


all_reviews = reviews_1_proc + reviews_4_proc + reviews_5_proc
labels = (
    ["1 estrela"] * len(reviews_1_proc) +
    ["4 estrelas"] * len(reviews_4_proc) +
    ["5 estrelas"] * len(reviews_5_proc)
)


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_reviews)


sim_matrix = cosine_similarity(embeddings)


df_sim = pd.DataFrame(sim_matrix, index=labels, columns=labels)
print("\nMatriz de Similaridade:")
print(df_sim.round(3))


def group_similarity(index_group1, index_group2):
    sims = []
    for i in index_group1:
        for j in index_group2:
            if i != j:  
                sims.append(sim_matrix[i][j])
    return np.mean(sims)


idx_1 = [0, 1, 2]
idx_2 = [3, 4, 5]
idx_3 = [6, 7, 8]


print("\nSimilaridade média entre grupos:")
print(f"1 estrela vs 4 estrelas: {group_similarity(idx_1, idx_2):.4f}")
print(f"1 estrela vs 5 estrelas: {group_similarity(idx_1, idx_3):.4f}")
print(f"4 estrelas vs 5 estrelas: {group_similarity(idx_2, idx_3):.4f}")