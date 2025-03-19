import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# 设置模型保存路径
model_save_path = Path("")

# 加载训练集数据
#train_data_path = r"C:\Users\Lenovo\Desktop\self-adaptive\traindata长庚.csv"
train_data_path = r""
train_data = pd.read_csv(train_data_path)

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5', cache_dir=model_save_path)
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5', cache_dir=model_save_path)

# 将模型移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 对finding列中的每条数据进行向量化
embeddings = []
sparse_embeddings = []
texts = []
labels = []
skipped_indices = []

# 稀疏向量是通过简单的词袋模型生成
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

vectorizer = CountVectorizer(max_features=1000)  # 选取1000个特征

# 文本收集进行稀疏向量生成
finding_texts = train_data['finding'].tolist()

# 生成稀疏向量
sparse_matrix = vectorizer.fit_transform(finding_texts)  # 不调用toarray()

for idx, row in train_data.iterrows():
    finding = row['finding']
    label = row['label']

    # Tokenize sentences
    encoded_input = tokenizer([finding], padding=True, truncation=True, max_length=512, return_tensors='pt')
    input_length = encoded_input['input_ids'].shape[1]

    if input_length > 512:
        print(f"跳过行 {idx}：输入长度 {input_length} 超过 512 tokens")
        skipped_indices.append(idx)
        embeddings.append(None)  # 插入一个None以保持索引一致
        continue

    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}  # 将输入数据移动到GPU

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embedding = model_output[0][:, 0]

    # normalize embeddings
    sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)
    embeddings.append(sentence_embedding.squeeze().tolist())
    texts.append(finding)
    labels.append(label)

# 创建稀疏向量
for idx in range(sparse_matrix.shape[0]):
    sparse_vector = sparse_matrix[idx]  # 在这里仍然保持稀疏格式
    # 获取非零元素的索引和值
    indices = sparse_vector.indices
    data = sparse_vector.data
    # 将非零元素转换为字典格式
    sparse_dict = {int(idx): float(val) for idx, val in zip(indices, data)}
    sparse_embeddings.append(sparse_dict)  # 添加到稀疏向量列表中

# 创建一个新的DataFrame来保存向量化的结果
embedding_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(len(embeddings[0]))])
sparse_embedding_df = pd.DataFrame(sparse_embeddings)  # 将字典转换为DataFrame

# 保存到新的CSV文件
#embedding_df.to_csv("C:/Users/Lenovo/Desktop/traindata肺部四分类_向量化长庚.csv", index=False)
#sparse_embedding_df.to_csv("C:/Users/Lenovo/Desktop/traindata肺部四分类_稀疏向量化长庚.csv", index=False)
embedding_df.to_csv(r"", index=False)
sparse_embedding_df.to_csv(r"", index=False)
print("向量化完成，结果已保存到新的CSV文件中。")
print(f"跳过的行数：{len(skipped_indices)}")

# 连接到Milvus数据库
connections.connect("default", host="localhost", port="19530")

# 定义集合名称
collection_name = "hybrid_demo1"  # 修改为以字母或下划线开头的名称

# 检查集合是否存在，如果存在则删除
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=100)
]

# 定义集合模式
schema = CollectionSchema(fields=fields, description="肺部四分类混合向量化数据友谊")

# 创建集合
collection = Collection(name=collection_name, schema=schema)

# 修改插入数据部分
for i in range(0, len(texts), 50):  # 每批处理50条数据
    batch_data = []
    for j in range(i, min(i + 50, len(texts))):
        entity = {
            "sparse_vector": sparse_embeddings[j],  # 已经是字典格式的稀疏向量
            "dense_vector": embeddings[j],
            "text": texts[j],
            "label": labels[j]
        }
        batch_data.append(entity)
    collection.insert(batch_data)



# 创建索引并加载集合
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
collection.create_index("sparse_vector", sparse_index)
dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
collection.create_index("dense_vector", dense_index)
collection.load()

print("数据插入和索引创建完成")