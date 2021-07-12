import jieba
import jieba.posseg as jp
import gensim
from gensim import corpora, models
import re
import sys
from gensim.models import TfidfModel,CoherenceModel
# Global Dictionary
new_words = ['新冠', '疫情']  # 新词
    # 获取停用词list
# stopwords.append('抱歉，作者已设置仅展示半年内微博，此微博已不可见。','O网页链接','收起全文d'])
synonyms = {'新冠': '新冠肺炎', '谣言': '不实信息'}  # 同义词
words_nature = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 可用的词性


def add_new_words():  # 增加新词
    for i in new_words:
        jieba.add_word(i)
#加载停用词
def stopwordslist(filepath):
     stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
     return stopwords
#去除停用词
def seg_sentence(sentence):
     sentence_seged = jieba.cut(sentence.strip())
     stopwords = stopwordslist('../scu_stopwords.txt')  # 这里加载停用词的路径
     outstr = ''
     for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                 outstr += word
                #outstr += " "
     return outstr

def clean(text):
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"\[\S+\]", "", text)      # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)       # 去除网址
    text = text.replace("转发微博", "")       # 去除无意义的词语
    text = text.replace("抱歉，作者已设置仅展示半年内微博，此微博已不可见。"," ")
    text = text.replace("该账号因被投诉违反《微博社区公约》的相关规定，现已无法查看。查看帮助"," ")
    text = re.sub(r"\s+", " ", text) # 合并正文中过多的空格
    return text.strip()


def replace_synonyms(ls):  # 替换同义词
    return [synonyms[i] if i in synonyms else i for i in ls]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#处理weibo数据
documents =[]
with open('./data/phase1.txt','r',encoding='utf-8') as f:
   content = f.readlines()
   for line in content:
       text = seg_sentence(clean(line))
       documents.append(text)
   # print('do:' + documents[7])

add_new_words()
words_ls = []
for text in documents:
    words = replace_synonyms([w.word for w in jp.cut(text)])#先jieba分词，去除停用词，然后同义词归一
    words_ls.append(words)
# 生成语料词典
dictionary = corpora.Dictionary(words_ls)
# 生成稀疏向量集
dictionary.filter_n_most_frequent(200)
corpus = [dictionary.doc2bow(words) for words in words_ls]
tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
# LDA模型，num_topics设置聚类数，即最终主题的数量
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3)

# 展示每个主题的前10的词语
for topic in lda.print_topics(num_words = 10):
    termNumber = topic[0]
    print(topic[0], ':', sep='')
    listOfTerms = topic[1].split('+')
    for term in listOfTerms:
        listItems = term.split('*')
        print('  ', listItems[1], '(', listItems[0], ')', sep='')
#         print("主题 {} : {}".format(topic_idx,"|".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])))
# 推断每个语料库中的主题类别
# """抽取新闻的主题"""
# corpus_test = [dictionary.doc2bow(text) for text in documents]
# # 得到每条新闻的主题分布
# topics_test = lda.get_document_topics(corpus_test)
# for i in range(3):
#     print('主题分布为：\n')
#     print(topics_test[i], '\n')
print('推断：')
savedStdout = sys.stdout
data = open('./results/phase1_lda.txt','w',encoding='utf-8')
sys.stdout = data
print('输出到文件')
for e, values in enumerate(lda.inference(corpus)[0]):
    topic_val = 0
    topic_id = 0
    for tid, val in enumerate(values):
        if val > topic_val:
            topic_val = val
            topic_id = tid
    # print(topic_id, '->', documents[e])
    # list=[topic_id, '->', documents[e]]
    print(topic_id, '->', documents[e])
    sys.stdout = savedStdout
    data.close()
    # print(topic_id, '->', documents[e],file = data)
    # with open('./results/phase1_lda.txt','w',encoding='utf-8') as fw:
    #     for str in list:
    #         fw.write(topic_id + '->' + documents[e])


