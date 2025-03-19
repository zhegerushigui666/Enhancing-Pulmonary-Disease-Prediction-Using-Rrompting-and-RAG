# -*- coding: utf-8 -*-
import time
import pandas as pd
from zhipuai import ZhipuAI
from tqdm import tqdm
from pymilvus import connections, Collection, FieldSchema, DataType, utility, CollectionSchema, AnnSearchRequest
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path
import jieba
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymilvus import AnnSearchRequest, WeightedRanker
from sklearn.feature_extraction.text import CountVectorizer
from pymilvus import RRFRanker
import logging
from openai import OpenAI



# 设置日志文件路径
log1_path = rf""
log2_path = rf""

# 设置日志格式和级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log2_path, mode='w', encoding='utf-8'),  # 保存所有打印输出到 log2
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

# 日志记录函数
def save_llm_log(content):
    with open(log1_path, "a", encoding="utf-8") as f:
        f.write(content + "\n")

rerank = RRFRanker(k=60)
# 初始化 CountVectorizer
vectorizer = CountVectorizer()

# 初始化 ZhipuAI 客户端和模型
#client = ZhipuAI(api_key="")  # 请替换为您的实际 API 密钥
#deepseek client
client = OpenAI(api_key="", base_url="https://api.deepseek.com")
#GPT client
#client = OpenAI(api_key="",base_url="")

model_save_path = Path(r"")
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5', cache_dir=model_save_path)
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5', cache_dir=model_save_path)

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 定义字段配置
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=100)
]

# 连接到 Milvus 服务器
connections.connect("default", host="localhost", port="19530")

# 检查是否存在集合，如果不存在则创建
collection_name = "hybrid_demo"
if not utility.has_collection(collection_name):
    collection_schema = CollectionSchema(fields, "肺部四分类混合向量化数据长庚")
    collection = Collection(collection_name, collection_schema)
#肺部四分类混合向量化数据友谊,肺部四分类混合向量化数据长庚
    # 创建 IVF_FLAT 索引
    index_params = {
        "index_type": "FLAT",
        "params": {"nlist": 128},
        "metric_type": "L2"
    }
    collection.create_index(field_name="embedding", index_params=index_params)
else:
    collection = Collection(collection_name)

# 加载集合
collection.load()

# 加载 CSV 并拟合 CountVectorizer
file_path = r""
#"C:\Users\Lenovo\Desktop\self-adaptive\test长庚.csv"
#"C:\Users\Lenovo\Desktop\self-adaptive\四分类测试集_source.csv"
df = pd.read_csv(file_path)
vectorizer.fit(df['imagefinding'].tolist())  # 拟合以便后续使用

# 定义生成文本嵌入的函数
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

# 混合检索函数
def hybrid_vector_search(finding):
    # 生成密集向量
    dense_embedding = get_embedding(finding)

    # 生成稀疏向量
    sparse_vector = vectorizer.transform([finding])
    indices = sparse_vector.indices
    data = sparse_vector.data
    sparse_dict = {int(idx): float(val) for idx, val in zip(indices, data)}

    # 创建搜索请求
    dense_request = AnnSearchRequest(
        data=[dense_embedding.tolist()],
        anns_field="dense_vector",
        param={
            "metric_type": "IP",  # 改为IP
            "params": {"nprobe": 10}
        },
        limit=50
    )

    sparse_request = AnnSearchRequest(
        data=[sparse_dict],
        anns_field="sparse_vector",
        param={
            "metric_type": "IP",
            "params": {"nprobe": 10}
        },
        limit=50
    )
    from pymilvus import RRFRanker


    # 初始化 BalancedRRFRanker
    rerank = RRFRanker(k=60)

    # Execute hybrid search
    search_results = collection.hybrid_search(
        reqs=[dense_request, sparse_request],
        rerank=rerank,  # Use rerank instead of ranker
        limit=30,
        output_fields=["text", "label"]
    )


    return search_results[0]

# 搜索和提取相似文本的函数
def search_and_extract(finding):
    hybrid_results = hybrid_vector_search(finding)
    examples = []
    used_labels = set()

    # 扩大检索结果的数量
    # hybrid_results = hybrid_results

    for hit in hybrid_results:
        text = hit.entity.get('text')
        label = hit.entity.get('label')  # 获取 label
        #question = hit.entity.get('question')
        # if text and label and label not in used_labels:  # 确保二者都存在且标签未被使用
        if text and label :  # 确保二者都存在
            examples.append({"  **影像报告**    ": text, "           **输出**         疾病诊断为": label})  # 使用字典来保存对应关系
            used_labels.add(label)
            if len(examples) >= 4:
                break

    # 如果不足4个，则取标签数的结果
    if len(examples) < 4:
        examples = []
        used_labels = set()
        for hit in hybrid_results:
            text = hit.entity.get('text')
            label = hit.entity.get('label')  # 获取 label
            #question = hit.entity.get('question')
            if text and label:  # 确保二者都存在且标签未被使用
                examples.append({"  **影像报告**    ": text, "          **输出**          疾病诊断为": label})  # 使用字典来保存对应关系
                used_labels.add(label)

    # 保存调试信息到日志
    logging.info(f"Finding: {finding} | Examples: {examples}")

    # print(examples)  # 调试输出
    return examples

def manual_prompt_CoT_RAG(finding, examples):
    response = client.chat.completions.create(
        model="glm-4-air",  # glm-4-air   glm-4-flash

        messages=[
            {"role": "user", "content": f"""
                        你是一位擅长肺部疾病诊断的医学专家，分析待诊断影像报告并回答问题，最终对患者进行肺炎、肺癌、肺结核或无病的诊断。请严格按照以下步骤和结构输出结果。

                        # 任务步骤


                        1. **参考真实病例诊断结果**  

                        {examples}
                        
                        2. **预测疾病类别**  
                        综合影像信息及问题分析结果，参考各疾病影像报告特点，预测患者可能的疾病类别。疾病类别仅限以下四种：肺炎、肺癌、肺结核、无病（“无病”特指无肺炎、肺癌、肺结核）。  

                        3. **结构化输出**  
                        严格按照指定格式输出诊断结果，不要输出无关内容。

                        # 输出格式

                        以下为诊断结果输出的标准格式：
                        ```
                        问题：
                        是否有钙化灶，或者钙化现象:(是/否)
                        是否出现了磨玻璃影，索条影，斑片影:(是/否)            
                        是否存在斑片状实变影或支气管血管束变化？ (是/否)
                        是否存在结节影或分叶征？ (是/否)
                        
                        --**综合判断**：
                        
                          - 基于以上特征，患者最可能的诊断为：____（肺结核 / 肺癌 / 肺炎 / 无病）。

                        ```

                        # 注意事项

                        - 请严格按照上述格式输出，不必输出思考过程。  
                        - 对无关疾病或症状无需进行标注。  
                        - 如影像报告中信息不足，请合理推断并基于已有信息输出结果。
                        - 当影像报告有症状出现时，要合理分析诊断为哪种疾病，对于无病的判断要谨慎。

                        待诊断影像报告：{finding}"""
             },
        ],
    )
    # 保存 LLM 的完整返回内容
    save_llm_log(response.choices[0].message.content)

    # 统计并记录 token 消耗
    # token_usage = response.usage.get("total_tokens", 0)
    token_usage = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
    logging.info(f"Finding: {finding} | Token Usage: {token_usage}")

    return response.choices[0].message.content



# 提取诊断印象的函数加完医生提到的特征，并使用了新的新的COT方法。
def FS_COT_RAG(finding, examples):
    response = client.chat.completions.create(
        model="deepseek-chat",  # glm-4-air   glm-4-flash glm-4-0520 glm-4-plus deepseek-chat gpt-4o gpt-4-turbo

        messages=[
            {"role": "user", "content": f"""
                        你是一位擅长肺部疾病诊断的医学专家，分析待诊断影像报告并回答问题，最终对患者进行肺炎、肺癌、肺结核或无病的诊断。请严格按照以下步骤和结构输出结果。

                        # 任务步骤

                        1. **学习各疾病影像报告特点（权重从高到低）**  

                        --钙化灶及结节影
                          肺结核: 斑点状、片状或结节状钙化灶，主要分布于上肺叶尖后段，伴纤维索条影与胸膜牵拉。
                          肺癌: 病灶内偶见钙化，通常呈边缘分叶状、伴毛刺征，钙化灶在肺癌中较为少见。
                          肺炎: 少见钙化灶，若存在也分布较分散，不伴有胸膜牵拉。
                          无病: 无钙化影或异常结节。

                        --空泡征与磨玻璃影
                          肺癌: 混合密度影（磨玻璃+实性），可见空泡征位于磨玻璃影中心或边缘，伴毛刺征和分叶。
                          肺结核: 可见薄壁空泡影，可能伴随有周围卫星灶，直径较大，伴斑片状实变和钙化灶。
                          肺炎: 散在分布磨玻璃密度影，边缘模糊，实变较常见，动态变化。
                          无病: 无空泡或磨玻璃影。

                        --实变影与支气管血管束变化
                          肺炎: 斑片状实变影常见，伴空气支气管征，病灶分布在下肺叶后基底段，不伴有支气管壁增厚。
                          肺结核: 斑片形态不规则实变，多在上肺叶，伴胸膜牵拉，可能伴随有支气管扩张。
                          肺癌: 实变影边界不规则，位于中肺叶或下肺叶，常与支气管闭塞相关。
                          无病: 支气管血管束清晰，无实变影。
                          
                        --胸膜增厚及粘连
                          肺结核: 常见局限性胸膜增厚，伴钙化或牵拉，可能与胸水并存。
                          肺癌: 局部胸膜增厚，与肿物相连，边界模糊，可能与肿瘤直接侵犯有关。
                          肺炎: 少见胸膜增厚，偶尔见轻微粘连，不伴有胸水。
                          无病: 胸膜结构正常。
                          
                        --结节与分叶征
                          肺癌: 结节密度不均，伴分叶征和血管集束征，是肺癌的重要特征。
                          肺结核: 结节边界清楚，内可见钙化，常为小叶中心结节，是结核的典型表现。
                          肺炎: 少见分叶征，结节较模糊，不如肺癌和肺结核明显。
                          无病: 无结节影。
                          
                        2. **参考真实病例诊断结果**  

                        {examples}


                        3. **预测疾病类别**  
                        综合影像信息及问题分析结果，预测患者可能的疾病类别。疾病类别仅限以下四种：肺炎、肺癌、肺结核、无病（“无病”特指无肺炎、肺癌、肺结核）。  

                        4. **结构化输出**  
                        严格按照指定格式输出诊断结果，不要输出无关内容。

                        # 输出格式

                        以下为诊断结果输出的标准格式：
                        ```
                        问题：
                        --1. **钙化灶及结节影**：
                          - 是否存在钙化灶或结节影？ [ 是 / 否 ]
                          - 若存在，钙化灶的分布特点为：____（如上肺叶尖后段分布、分散分布等）。

                        --2. **空泡征与磨玻璃影**：
                          - 是否存在空泡征或磨玻璃影？ [ 是 / 否 ]
                          - 若存在，空泡征的位置和磨玻璃影的特征为：____（如位于中心或边缘、密度分布等）。

                        --3. **实变影与支气管血管束变化**：
                          - 是否存在斑片状实变影或支气管血管束变化？ [ 是 / 否 ]
                          - 若存在，实变影的分布部位和支气管血管束的变化情况为：____（如下肺叶后基底段、支气管闭塞等）。

                        --4. **胸膜增厚及粘连**：
                          - 是否存在胸膜增厚或粘连？ [ 是 / 否 ]
                          - 若存在，胸膜增厚的部位及是否伴有钙化或牵拉为：____。

                        --5. **结节与分叶征**：
                          - 是否存在结节影或分叶征？ [ 是 / 否 ]
                          - 若存在，结节的密度分布及分叶征的明显程度为：____。

                        --**综合判断**：
                          - 基于以上特征，患者最可能的诊断为：____（肺结核 / 肺癌 / 肺炎 / 无病）。

                        ```

                        # 注意事项

                        - 请严格按照上述格式输出，不必输出思考过程。  
                        - 对无关疾病或症状无需进行标注。  
                        - 如影像报告中信息不足，请合理推断并基于已有信息输出结果。
                        - 对于无病的判断要谨慎，有症状存在的情况，不要错把患病当作无病。

                        待诊断影像报告：{finding}"""
             },
        ],
    )
    # 保存 LLM 的完整返回内容
    save_llm_log(response.choices[0].message.content)

    # 统计并记录 token 消耗
    # token_usage = response.usage.get("total_tokens", 0)
    token_usage = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
    logging.info(f"Finding: {finding} | Token Usage: {token_usage}")

    return response.choices[0].message.content




def FS_fewshot(finding, examples):
    response = client.chat.completions.create(
        model="deepseek-chat",  # glm-4-air   glm-4-flash deepseek-chat

        messages=[
            {"role": "user", "content": f"""
                        你是一位擅长肺部疾病诊断的医学专家，分析待诊断影像报告并回答问题，最终对患者进行肺炎、肺癌、肺结核或无病的诊断。请严格按照以下步骤和结构输出结果。

                        # 任务步骤

                        1. **学习各疾病影像报告特点（权重从高到低）**  

                        --钙化灶及结节影
                          肺结核: 斑点状、片状或结节状钙化灶，主要分布于上肺叶尖后段，伴纤维索条影与胸膜牵拉。
                          肺癌: 病灶内偶见钙化，通常呈边缘分叶状、伴毛刺征，钙化灶在肺癌中较为少见。
                          肺炎: 少见钙化灶，若存在也分布较分散，不伴有胸膜牵拉。
                          无病: 无钙化影或异常结节。

                        --空泡征与磨玻璃影
                          肺癌: 混合密度影（磨玻璃+实性），可见空泡征位于磨玻璃影中心或边缘，伴毛刺征和分叶。
                          肺结核: 可见薄壁空泡影，可能伴随有周围卫星灶，直径较大，伴斑片状实变和钙化灶。
                          肺炎: 散在分布磨玻璃密度影，边缘模糊，实变较常见，动态变化。
                          无病: 无空泡或磨玻璃影。

                        --实变影与支气管血管束变化
                          肺炎: 斑片状实变影常见，伴空气支气管征，病灶分布在下肺叶后基底段，不伴有支气管壁增厚。
                          肺结核: 斑片形态不规则实变，多在上肺叶，伴胸膜牵拉，可能伴随有支气管扩张。
                          肺癌: 实变影边界不规则，位于中肺叶或下肺叶，常与支气管闭塞相关。
                          无病: 支气管血管束清晰，无实变影。

                        --胸膜增厚及粘连
                          肺结核: 常见局限性胸膜增厚，伴钙化或牵拉，可能与胸水并存。
                          肺癌: 局部胸膜增厚，与肿物相连，边界模糊，可能与肿瘤直接侵犯有关。
                          肺炎: 少见胸膜增厚，偶尔见轻微粘连，不伴有胸水。
                          无病: 胸膜结构正常。

                        --结节与分叶征
                          肺癌: 结节密度不均，伴分叶征和血管集束征，是肺癌的重要特征。
                          肺结核: 结节边界清楚，内可见钙化，常为小叶中心结节，是结核的典型表现。
                          肺炎: 少见分叶征，结节较模糊，不如肺癌和肺结核明显。
                          无病: 无结节影。

                        2. **参考真实病例诊断结果**  

                        finding：

                          胸廓两侧对称，支气管血管束清晰。右肺中叶及左肺上叶舌段可见索条实变影及磨玻璃密度影。双肺下叶可见少许磨玻璃密度影。主气管、双肺支气管及其分支管腔通畅。双侧肺门及纵隔内多发淋巴结，部分稍大，大者短径约1.1cm。心脏增大。双侧部分胸膜增厚。 增强后，主动脉及冠状动脉管壁散在钙化斑块。肺动脉主干及左右干其分支造影剂充盈良好，双肺下叶部分肺动脉分支内密度不均。 肝实质密度弥漫稍低。


                          基于以上文本，患者最可能的诊断为：肺炎。


                        3. **预测疾病类别**  
                        综合影像信息及问题分析结果，预测患者可能的疾病类别。疾病类别仅限以下四种：肺炎、肺癌、肺结核、无病（“无病”特指无肺炎、肺癌、肺结核）。  

                        4. **结构化输出**  
                        严格按照指定格式输出诊断结果，不要输出无关内容。

                        # 输出格式

                        以下为诊断结果输出的标准格式：
                        ```
                        
                          - 基于以上文本，患者最可能的诊断为：____（肺结核 / 肺癌 / 肺炎 / 无病）。

                        ```

                        # 注意事项

                        - 请严格按照上述格式输出，不必输出思考过程。  
                        - 对无关疾病或症状无需进行标注。  
                        - 如影像报告中信息不足，请合理推断并基于已有信息输出结果。
                        - 对于无病的判断要谨慎，有症状存在的情况，不要错把患病当作无病。

                        待诊断影像报告：{finding}


"""
             },
        ],
    )
    # 保存 LLM 的完整返回内容
    save_llm_log(response.choices[0].message.content)

    # 统计并记录 token 消耗
    # token_usage = response.usage.get("total_tokens", 0)
    token_usage = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
    logging.info(f"Finding: {finding} | Token Usage: {token_usage}")

    return response.choices[0].message.content
# 提取诊断印象的函数
def FS_fewshot_COT(finding, examples):
    response = client.chat.completions.create(
        model="deepseek-chat",  # glm-4-air   glm-4-flash deepseek-chat

        messages=[
            {"role": "user", "content": f"""
                        你是一位擅长肺部疾病诊断的医学专家，分析待诊断影像报告并回答问题，最终对患者进行肺炎、肺癌、肺结核或无病的诊断。请严格按照以下步骤和结构输出结果。

                        # 任务步骤

                        1. **学习各疾病影像报告特点（权重从高到低）**  

                        --钙化灶及结节影
                          肺结核: 斑点状、片状或结节状钙化灶，主要分布于上肺叶尖后段，伴纤维索条影与胸膜牵拉。
                          肺癌: 病灶内偶见钙化，通常呈边缘分叶状、伴毛刺征，钙化灶在肺癌中较为少见。
                          肺炎: 少见钙化灶，若存在也分布较分散，不伴有胸膜牵拉。
                          无病: 无钙化影或异常结节。

                        --空泡征与磨玻璃影
                          肺癌: 混合密度影（磨玻璃+实性），可见空泡征位于磨玻璃影中心或边缘，伴毛刺征和分叶。
                          肺结核: 可见薄壁空泡影，可能伴随有周围卫星灶，直径较大，伴斑片状实变和钙化灶。
                          肺炎: 散在分布磨玻璃密度影，边缘模糊，实变较常见，动态变化。
                          无病: 无空泡或磨玻璃影。

                        --实变影与支气管血管束变化
                          肺炎: 斑片状实变影常见，伴空气支气管征，病灶分布在下肺叶后基底段，不伴有支气管壁增厚。
                          肺结核: 斑片形态不规则实变，多在上肺叶，伴胸膜牵拉，可能伴随有支气管扩张。
                          肺癌: 实变影边界不规则，位于中肺叶或下肺叶，常与支气管闭塞相关。
                          无病: 支气管血管束清晰，无实变影。
                          
                        --胸膜增厚及粘连
                          肺结核: 常见局限性胸膜增厚，伴钙化或牵拉，可能与胸水并存。
                          肺癌: 局部胸膜增厚，与肿物相连，边界模糊，可能与肿瘤直接侵犯有关。
                          肺炎: 少见胸膜增厚，偶尔见轻微粘连，不伴有胸水。
                          无病: 胸膜结构正常。
                          
                        --结节与分叶征
                          肺癌: 结节密度不均，伴分叶征和血管集束征，是肺癌的重要特征。
                          肺结核: 结节边界清楚，内可见钙化，常为小叶中心结节，是结核的典型表现。
                          肺炎: 少见分叶征，结节较模糊，不如肺癌和肺结核明显。
                          无病: 无结节影。
                          
                        2. **参考真实病例诊断结果**  

                        finding：

                          胸廓两侧对称，支气管血管束清晰。右肺中叶及左肺上叶舌段可见索条实变影及磨玻璃密度影。双肺下叶可见少许磨玻璃密度影。主气管、双肺支气管及其分支管腔通畅。双侧肺门及纵隔内多发淋巴结，部分稍大，大者短径约1.1cm。心脏增大。双侧部分胸膜增厚。 增强后，主动脉及冠状动脉管壁散在钙化斑块。肺动脉主干及左右干其分支造影剂充盈良好，双肺下叶部分肺动脉分支内密度不均。 肝实质密度弥漫稍低。

                          问题：
                          --1. 钙化灶及结节影：

                          是否存在钙化灶或结节影？ [ 否 ]
                          若存在，钙化灶的分布特点为：____。
                          --2. 空泡征与磨玻璃影：

                          是否存在空泡征或磨玻璃影？ [ 是 ]
                          若存在，空泡征的位置和磨玻璃影的特征为：右肺中叶及左肺上叶舌段可见索条实变影及磨玻璃密度影，双肺下叶可见少许磨玻璃密度影。
                          --3. 实变影与支气管血管束变化：

                          是否存在斑片状实变影或支气管血管束变化？ [ 是 ]
                          若存在，实变影的分布部位和支气管血管束的变化情况为：右肺中叶及左肺上叶舌段可见索条实变影。
                          --4. 胸膜增厚及粘连：

                          是否存在胸膜增厚或粘连？ [ 是 ]
                          若存在，胸膜增厚的部位及是否伴有钙化或牵拉为：双侧部分胸膜增厚。
                          --5. 结节与分叶征：

                          是否存在结节影或分叶征？ [ 否 ]
                          若存在，结节的密度分布及分叶征的明显程度为：____。
                          --综合判断：

                          基于以上特征，患者最可能的诊断为：肺炎。


                        3. **预测疾病类别**  
                        综合影像信息及问题分析结果，预测患者可能的疾病类别。疾病类别仅限以下四种：肺炎、肺癌、肺结核、无病（“无病”特指无肺炎、肺癌、肺结核）。  

                        4. **结构化输出**  
                        严格按照指定格式输出诊断结果，不要输出无关内容。

                        # 输出格式

                        以下为诊断结果输出的标准格式：
                        ```
                        问题：
                        --1. **钙化灶及结节影**：
                          - 是否存在钙化灶或结节影？ [ 是 / 否 ]
                          - 若存在，钙化灶的分布特点为：____（如上肺叶尖后段分布、分散分布等）。

                        --2. **空泡征与磨玻璃影**：
                          - 是否存在空泡征或磨玻璃影？ [ 是 / 否 ]
                          - 若存在，空泡征的位置和磨玻璃影的特征为：____（如位于中心或边缘、密度分布等）。

                        --3. **实变影与支气管血管束变化**：
                          - 是否存在斑片状实变影或支气管血管束变化？ [ 是 / 否 ]
                          - 若存在，实变影的分布部位和支气管血管束的变化情况为：____（如下肺叶后基底段、支气管闭塞等）。

                        --4. **胸膜增厚及粘连**：
                          - 是否存在胸膜增厚或粘连？ [ 是 / 否 ]
                          - 若存在，胸膜增厚的部位及是否伴有钙化或牵拉为：____。

                        --5. **结节与分叶征**：
                          - 是否存在结节影或分叶征？ [ 是 / 否 ]
                          - 若存在，结节的密度分布及分叶征的明显程度为：____。

                        --**综合判断**：
                          - 基于以上特征，患者最可能的诊断为：____（肺结核 / 肺癌 / 肺炎 / 无病）。

                        ```

                        # 注意事项

                        - 请严格按照上述格式输出，不必输出思考过程。  
                        - 对无关疾病或症状无需进行标注。  
                        - 如影像报告中信息不足，请合理推断并基于已有信息输出结果。
                        - 对于无病的判断要谨慎，有症状存在的情况，不要错把患病当作无病。

                        待诊断影像报告：{finding}


"""
            },
        ],
    )
    # 保存 LLM 的完整返回内容
    save_llm_log(response.choices[0].message.content)

    # 统计并记录 token 消耗
    # token_usage = response.usage.get("total_tokens", 0)
    token_usage = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
    logging.info(f"Finding: {finding} | Token Usage: {token_usage}")

    return response.choices[0].message.content


# 提取诊断印象的函数



# 处理 CSV 文件的函数
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_csv_file(file_path):
    # 读取原始文件
    df = pd.read_csv(file_path)

    # 提取所有 findings 和真实标签
    findings = df['imagefinding'].tolist()
    real_labels = df['label'].tolist()

    all_examples = []
    predicted_labels = []

    # 使用多线程处理每个 finding
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(search_and_extract, findings))
        all_examples.extend(results)

        for idx, examples in enumerate(results):
            impression = FS_COT_RAG(findings[idx], examples)
            predicted_labels.append(impression)
            df.loc[idx, 'predict label'] = impression

            # 记录日志
            logging.info(f"Processed finding: {findings[idx]} | Impression: {impression}")

    # 保存处理后的 CSV 文件
    output_csv = r""
    save_to_unified_csv(findings, all_examples, predicted_labels, real_labels, output_csv)

    logging.info(f"Processed findings saved to: {output_csv}")



# 在 process_csv_file 的最后添加保存到 Excel 的逻辑

def save_to_unified_csv(findings, examples_list, predicted_labels, real_labels, output_path):
    data = []

    # 整合每个 finding 的结果
    for finding, examples, predicted_label, real_label in zip(findings, examples_list, predicted_labels, real_labels):
        data.append({
            "Finding": finding,
            "Example Text": examples,
            "Predicted Label": predicted_label,
            "Real Label": real_label
        })        

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 保存到 CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"Unified data saved to CSV: {output_path}")



# 文件路径
file_path = r""
process_csv_file(file_path)

