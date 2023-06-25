import os
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import BartTokenizer, BartForConditionalGeneration


import networkx as nx
import matplotlib.pyplot as plt



nltk.download('punkt')

# 載入GloVe詞向量模型
# 6B: 使用60億的單字語料庫訓練，300d代表300維度
# 檔案的第一個欄位是單字，第二欄開始為詞向量
# 較高的維度可以提供更多的語義資訊
def load_word_vectors(file_path):
    word_vectors = {} # 用來儲存詞向量
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split() # 每一行以空格進行分割
            word = values[0] # 取得單字
            vector = np.asarray(values[1:], dtype='float32') # 詞向量
            word_vectors[word] = vector
    print(f"king: {word_vectors['king']}")

    return word_vectors


"""
np.mean範例:

# 假設有一個數組
arr = [1, 2, 3, 4, 5]

# 使用 np.mean 函數計算平均值
mean_value = np.mean(arr)

print(mean_value)  # 輸出結果為 3.0
"""




def sentence_to_vector(sentence, word_vectors):
    words = sentence.split()
    print(f'words: {words}')
    # 獲取單字的詞向量，如果單字不存在則返回一個300維的0向量
    vectors = [word_vectors.get(word.lower(), np.zeros(300)) for word in words]
    # print(f'vectors: {vectors}')
    if len(vectors) > 0:
        sentence_vector = np.mean(vectors, axis=0)
        # print(f'sentence_vector: {sentence_vector}')
    else:
        sentence_vector = np.zeros(300)
    return sentence_vector




def compute_similarity_matrix(sentence_vectors):
    num_sentences = len(sentence_vectors)
    similarity_matrix = np.zeros((num_sentences, num_sentences))

    for i in range(num_sentences):
        for j in range(num_sentences):
            if i == j:
                similarity_matrix[i, j] = 1.0  # 對角線上的值設為1，表示與自身相比相似度最高，也就是同一個句子跟自己比較是一樣的
            else:
                vector_i = sentence_vectors[i]
                vector_j = sentence_vectors[j]
                similarity = np.dot(vector_i, vector_j) / (np.linalg.norm(vector_i) * np.linalg.norm(vector_j))
                similarity_matrix[i, j] = similarity

    return similarity_matrix


def textrank(similarity_matrix, d=0.85, max_iter=100, tol=1e-4):
    num_sentences = similarity_matrix.shape[0]
    scores = np.ones(num_sentences)  # 初始化權重為1
    old_scores = np.zeros(num_sentences)
    sentence_scores = {} # 儲存句子分數

    for _ in range(max_iter):
        scores_difference = np.sum(np.abs(scores - old_scores))
        if scores_difference < tol:
            break

        old_scores = scores.copy()

        for i in range(num_sentences):
            # similarity_matrix[:, i]: 以當前句子為目標的所有相似度
            # np.nonzero(similarity_matrix[:, i]): 找出所有相似度不為0的句子
            incoming_edges = np.nonzero(similarity_matrix[:, i])[0]  
            if len(incoming_edges) > 0:
                weights = similarity_matrix[incoming_edges, i] / np.sum(similarity_matrix[incoming_edges, :], axis=1)  # 計算權重
                scores[i] = (1 - d) + d * np.sum(weights * old_scores[incoming_edges])  # 使用TextRank公式更新權重
                sentence_scores[i] = scores[i]

    return scores, sentence_scores


def bart_summary(text):
    # print(f'text: {text}')
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    print(f'max_length: {tokenizer.model_max_length}')
    # BartForConditionalGeneration可以用於生成摘要
    # from_pretrained:載入預訓練模型
    model = BartForConditionalGeneration.from_pretrained(model_name)
    # 將list內的元素以空格連接起來，變成一個字串
    combined_string = ' '.join(text)
    text = combined_string
    # print(f'text: {text}')
    # truncation: 表示在使用BART tokenizer對文本進行Encode時，如果文本長度超過max_length，則自動進行截斷
    # 如果 padding='longest': 確保所有文本具有相同的長度，方便模型並行計算
    # return_tensors='pt': 返回PyTorch tensors，可以方便地用於BART模型的輸入，而不需要額外的轉換或處理
    encoded_input = tokenizer(text, truncation=True, padding='do_not_pad', max_length=1024, return_tensors='pt')
    # print(f'encoded_input {encoded_input}')
    # encoded_input['input_ids']: 經過BART tokenizer處理後的文本，轉換成數字序列，每個token都有一個對應的ID
    # beam search: 生成文本的一種算法，較大的值增加生成文本的多樣性，但會降低生成文本的品質(範圍通常是1~10)
    # max_length=500: 生成的摘要最大長度為500
    # early_stopping=True: 當達到最大長度限制時提前停止生成摘要
    summary_ids = model.generate(encoded_input['input_ids'], num_beams=4, max_length=500, early_stopping=True)
    # print(f'summary_ids: {summary_ids}')
    # summary_ids.squeeze(): 用於刪除在張量中任何多餘的維度
    # skip_special_tokens=True: 讓Decoder忽略特殊token，只返回具有意義的token，來生成最終的摘要
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return summary


def graph(sentences, similarity_matrix):
    # 創建無向圖
    G = nx.Graph()
    # 添加節點
    for i in range(len(sentences)):
        G.add_node(i)
    # 添加邊
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            similarity = similarity_matrix[i, j]
            G.add_edge(i, j, weight=similarity)

    # 繪製節點圖
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.title('Similarity Graph')
    plt.axis('off')
    plt.show()



def main():
    
    # sentence tokenization
    article = '''(CNN)Anthony Ray Hinton is thankful to be free after nearly 30 years on Alabama's death row for murders he says he didn't commit. And incredulous that it took so long. Hinton, 58, looked up, took in the sunshine and thanked God and his lawyers Friday morning outside the county jail in Birmingham, minutes after taking his first steps as a free man since 1985. He spoke of unjustly losing three decades of his life, under fear of execution, for something he didn't do. "All they had to do was to test the gun, but when you think you're high and mighty and you're above the law, you don't have to answer to nobody," Hinton told reporters. "But I've got news for you -- everybody that played a part in sending me to death row, you will answer to God." Jefferson County Circuit Court Judge Laura Petro had ordered Hinton released after granting the state's motion to dismiss charges against him. Hinton was convicted of murder in the 1985 deaths of two Birmingham-area, fast-food restaurant managers, John Davidson and Thomas Wayne Vason. But a new trial was ordered in 2014 after firearms experts testified 12 years earlier that the revolver Hinton was said to have used in the crimes could not be matched to evidence in either case, and the two killings couldn't be linked to each other. "Death Row Stories": Hard questions about the U.S. capital punishment system . The state then declined to re-prosecute the case. Hinton was 29 at the time of the killings and had always maintained his innocence, said the Equal Justice Initiative, a group that helped win his release. "Race, poverty, inadequate legal assistance, and prosecutorial indifference to innocence conspired to create a textbook example of injustice," Bryan Stevenson, the group's executive director and Hinton's lead attorney, said of his African-American client. "I can't think of a case that more urgently dramatizes the need for reform than what has happened to Anthony Ray Hinton." Stevenson said the "refusal of state prosecutors to re-examine this case despite persuasive and reliable evidence of innocence is disappointing and troubling." Amnesty report: Executions down but death sentences on the rise . Dressed in a dark suit and blue shirt, Hinton praised God for his release, saying he was sent "not just a lawyer, but the best lawyers." He said he will continue to pray for the families of the murder victims. Both he and those families have suffered a miscarriage of justice, he said. "For all of us that say that we believe in justice, this is the case to start showing, because I shouldn't have (sat) on death row for 30 years," he said. Woman who spent 22 years on death row has case tossed . Hinton was accompanied Friday by two of his sisters, one of whom still lives in the Birmingham area. Other siblings will fly to the area to see him soon, Stevenson said. His mother, with whom he lived at the time of his arrest, is no longer living, according to the lawyer. Hinton planned to spend at least this weekend at the home of a close friend. He will meet with his attorneys Monday to start planning for his immediate needs, such as obtaining identification and getting a health checkup, Stevenson said. The plan now is to spend a few weeks to get oriented with freedom and "sort out what he wants to do," Stevenson said.'''
    sentences = sent_tokenize(article)
    print(f'總共有: {len(sentences)} 個句子')

    folder_path = "glove/glove.6B"
    glove_file = 'glove.6B.300d.txt'
    glove_file = os.path.abspath(os.path.join(folder_path, glove_file))

    # load GloVe word vectors
    word_vectors = load_word_vectors(glove_file)
    # print(word_vectors)

    sentence_vectors = []
    for sentence in sentences:
        vector = sentence_to_vector(sentence, word_vectors)
        sentence_vectors.append(vector)

    print(len(sentence_vectors))

    similarity_matrix = compute_similarity_matrix(sentence_vectors)
    print(f'similarity_matrix: {similarity_matrix}')

    sentence_scores, summary_scores = textrank(similarity_matrix)
    # 用-降序排列
    sorted_indices = np.argsort(-sentence_scores)
    # 取前10%的句子
    num_sentences= round(len(sentences) * 0.1)
    #num_sentences = 27
    top_indices = sorted_indices[:num_sentences]
    textrank_summary = [sentences[i] for i in top_indices]

    for i, sentence in enumerate(textrank_summary):
        score = summary_scores[top_indices[i]]
        print(f"句子: {sentence}")
        print(f"TextRank的分數: {score}")


    bart = bart_summary(sentences)
    print(f'bart: {bart}')

    bart = sent_tokenize(bart)

    final_summary = bart_summary(bart)

    merged_summary = list(set(textrank_summary + bart))
    print(f'merged_summary: {merged_summary}')

    print(f'final_summary: {final_summary}')



    # graph(sentences, similarity_matrix)


    # sentence_level_word_embedding(article)


if __name__ == '__main__':
    main()