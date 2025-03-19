# -*- coding: utf-8 -*-
import time
import pandas as pd
from pymilvus import (
    connections, Collection, FieldSchema, DataType,
    utility, CollectionSchema, AnnSearchRequest, RRFRanker
)
from transformers import AutoTokenizer, AutoModel
import torch
import logging
from sklearn.feature_extraction.text import CountVectorizer
from openai import OpenAI
from pathlib import Path
import os
import json

# ============ 配置部分 ============
class Config:
    # Milvus 配置
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"
    COLLECTION_NAME = "hybrid_demo1"

    # 模型配置
    DENSE_DIM = 1024
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_save_path = Path(r"D:\model临时")

    # 文件路径
    CSV_PATH = r"C:\Users\Lenovo\Desktop\self-adaptive\四分类测试集_source.csv"
    OUTPUT_CSV = r"C:\Users\Lenovo\Desktop\self-adaptive\友谊测试集数据集新FS_COT_RAG特征排序deepseek_2024391706.csv"
    LOG_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "大模型RAG肺部疾病相关数据", "log_deepseek_38442.txt")
    RETRIEVAL_RESULTS_PATH = r"C:\Users\Lenovo\Desktop\self-adaptive"


# ============ 初始化部分 ============
def save_llm_log(content):
    with open(Config.LOG_PATH, "a", encoding="utf-8") as f:
        f.write(content + "\n")


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_PATH, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 初始化模型
model = AutoModel.from_pretrained(r"D:\model临时\BAAI\bge-large-zh-v1___5").to(Config.DEVICE)
tokenizer = AutoTokenizer.from_pretrained(r"D:\model临时\BAAI\bge-large-zh-v1___5")
model.eval()

# 初始化稀疏编码器
vectorizer = CountVectorizer()
df = pd.read_csv(Config.CSV_PATH)
vectorizer.fit(df['imagefinding'].tolist())

# 初始化Milvus
connections.connect("default", host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)


# ============ Milvus集合管理 ============
def setup_collection():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=Config.DENSE_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=100)
    ]

    if not utility.has_collection(Config.COLLECTION_NAME):
        schema = CollectionSchema(fields, "肺部四分类混合检索数据集友谊")
        collection = Collection(Config.COLLECTION_NAME, schema)

        # 创建密集向量索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 128}
        }
        collection.create_index("dense_vector", index_params)
        logging.info("Collection created with index")
    else:
        collection = Collection(Config.COLLECTION_NAME)

    collection.load()
    return collection


collection = setup_collection()


# ============ 核心检索逻辑 ============
class VectorSearcher:
    def __init__(self, mode='hybrid', rrf_k=60):
        """
        检索模式配置:
        - mode: hybrid/dense/sparse
        - rrf_k: RRF算法参数
        """
        self.mode = mode
        self.rrf_k = rrf_k
        self.search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

    def generate_vectors(self, text):
        """生成双路向量"""
        # 密集向量
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(Config.DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        dense_vec = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()

        # 稀疏向量
        sparse_vec = vectorizer.transform([text])
        sparse_dict = {int(i): float(v) for i, v in zip(sparse_vec.indices, sparse_vec.data)}

        return dense_vec, sparse_dict

    def search(self, text, top_k=4):
        """执行检索"""
        dense_vec, sparse_dict = self.generate_vectors(text)
        reqs = []

        # 构建搜索请求
        if self.mode in ('hybrid', 'dense'):
            reqs.append(AnnSearchRequest(
                data=[dense_vec],
                anns_field="dense_vector",
                param=self.search_params,
                limit=top_k * 2  # 扩大召回
            ))

        if self.mode in ('hybrid', 'sparse'):
            reqs.append(AnnSearchRequest(
                data=[sparse_dict],
                anns_field="sparse_vector",
                param=self.search_params,
                limit=top_k * 2
            ))

        # 执行搜索
        if self.mode == 'hybrid':
            results = collection.hybrid_search(
                reqs=reqs,
                rerank=RRFRanker(k=self.rrf_k),
                limit=top_k,
                output_fields=["text", "label"]
            )
        else:
            results = collection.search(
                data=[dense_vec] if self.mode == 'dense' else [sparse_dict],
                anns_field="dense_vector" if self.mode == 'dense' else "sparse_vector",
                param=self.search_params,
                limit=top_k,
                output_fields=["text", "label"]
            )

        return self.process_results(results[0], top_k)

    def process_results(self, results, top_k):
        """结果后处理（包含标签平衡功能）"""
        unique_labels = set()
        final_results = []

        # 第一轮筛选：确保每个标签只出现一次
        for hit in results:
            if len(final_results) >= top_k:
                break

            text = hit.entity.get('text')
            label = hit.entity.get('label')

            if text and label and label not in unique_labels:
                final_results.append({
                    "text": text,
                    "label": label,
                    "score": hit.score
                })
                unique_labels.add(label)

        # 如果结果不足，放宽标签限制
        if len(final_results) < top_k:
            for hit in results:
                if len(final_results) >= top_k:
                    break

                text = hit.entity.get('text')
                label = hit.entity.get('label')

                if text and label:
                    final_results.append({
                        "text": text,
                        "label": label,
                        "score": hit.score
                    })

        return final_results[:top_k]


# ============ 检索阶段 ============
def retrieve_all_queries(csv_path, output_dir):
    """检索所有查询并保存结果（每个检索方法独立存储）"""
    df = pd.read_csv(csv_path)
    modes = ['sparse', 'dense', 'hybrid']

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    for mode in modes:
        all_results = {}
        searcher = VectorSearcher(mode=mode)

        for idx, row in df.iterrows():
            finding = row['imagefinding']
            try:
                # 执行检索
                results = searcher.search(finding)
                all_results[idx] = results
                logging.info(f"{mode.upper()}模式: 检索完成 case {idx + 1}/{len(df)}")
            except Exception as e:
                logging.error(f"{mode.upper()}模式: 检索失败 case {idx}: {str(e)}")

        # 保存检索结果到文件
        output_path = os.path.join(output_dir, f"{mode}_results.json")  # 保存到指定目录
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        logging.info(f"{mode.upper()}模式: 所有结果已保存到 {output_path}")

# ============ API 调用阶段 ============
def process_results_with_api(output_dir, output_path):
    """基于检索结果调用 API 并保存最终结果"""
    df = pd.read_csv(Config.CSV_PATH)
    modes = ['sparse', 'dense', 'hybrid']

    for mode in modes:
        # 构建文件路径
        results_path = os.path.join(output_dir, f"{mode}_results.json")

        # 检查文件是否存在
        if not os.path.exists(results_path):
            logging.error(f"{mode.upper()}模式: 检索结果文件不存在 {results_path}")
            continue

        # 加载检索结果
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # 调用 API 生成诊断
        for idx, result in results.items():
            try:
                diagnosis = FS_COT_RAG(finding=df.loc[int(idx), 'imagefinding'], examples=result)
                df.loc[int(idx), f'{mode}_predicted_label'] = diagnosis
                df.loc[int(idx), f'{mode}_retrieved_examples'] = str(result)
                logging.info(f"{mode.upper()}模式: 处理完成 case {int(idx) + 1}/{len(results)}")
            except Exception as e:
                logging.error(f"{mode.upper()}模式: 处理失败 case {idx}: {str(e)}")

    # 保存最终结果到文件
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"最终结果已保存到 {output_path}")


def FS_COT_RAG(finding, examples):
    """调用 API 生成诊断"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": f"""
                        你是一位擅长肺部疾病诊断的医学专家，分析待诊断影像报告并回答问题，最终对患者进行肺炎、肺癌、肺结核或无病的诊断。请严格按照以下步骤和结构输出结果。

        # 任务步骤

        1. ** 学习各疾病影像报告特点（权重从高到低） **

        --钙化灶及结节影
        肺结核: 斑点状、片状或结节状钙化灶，主要分布于上肺叶尖后段，伴纤维索条影与胸膜牵拉。
        肺癌: 病灶内偶见钙化，通常呈边缘分叶状、伴毛刺征，钙化灶在肺癌中较为少见。
        肺炎: 少见钙化灶，若存在也分布较分散，不伴有胸膜牵拉。
        无病: 无钙化影或异常结节。

        --空泡征与磨玻璃影
        肺癌: 混合密度影（磨玻璃 + 实性），可见空泡征位于磨玻璃影中心或边缘，伴毛刺征和分叶。
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

        2. ** 参考真实病例诊断结果 **

        {examples}

        3. ** 预测疾病类别 **
        综合影像信息及问题分析结果，预测患者可能的疾病类别。疾病类别仅限以下四种：肺炎、肺癌、肺结核、无病（“无病”特指无肺炎、肺癌、肺结核）。

        4. ** 结构化输出 **
        严格按照指定格式输出诊断结果，不要输出无关内容。

        # 输出格式

        以下为诊断结果输出的标准格式：
        ```
        问题：
        --1. ** 钙化灶及结节影 **：
        - 是否存在钙化灶或结节影？ [是 / 否]
        - 若存在，钙化灶的分布特点为：____（如上肺叶尖后段分布、分散分布等）。

        --2. ** 空泡征与磨玻璃影 **：
        - 是否存在空泡征或磨玻璃影？ [是 / 否]
        - 若存在，空泡征的位置和磨玻璃影的特征为：____（如位于中心或边缘、密度分布等）。

        --3. ** 实变影与支气管血管束变化 **：
        - 是否存在斑片状实变影或支气管血管束变化？ [是 / 否]
        - 若存在，实变影的分布部位和支气管血管束的变化情况为：____（如下肺叶后基底段、支气管闭塞等）。

        --4. ** 胸膜增厚及粘连 **：
        - 是否存在胸膜增厚或粘连？ [是 / 否]
        - 若存在，胸膜增厚的部位及是否伴有钙化或牵拉为：____。

        --5. ** 结节与分叶征 **：
        - 是否存在结节影或分叶征？ [是 / 否]
        - 若存在，结节的密度分布及分叶征的明显程度为：____。

        -- ** 综合判断 **：
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
    token_usage = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
    logging.info(f"Finding: {finding} | Token Usage: {token_usage}")

    return response.choices[0].message.content

# ============ 辅助函数 ============
def check_retrieval_files_exist(output_dir):
    """检查检索结果文件是否存在"""
    modes = ['sparse', 'dense', 'hybrid']
    for mode in modes:
        file_path = os.path.join(output_dir, f"{mode}_results.json")
        logging.info(f"检查文件路径: {file_path}")  # 输出实际检查的路径
        if not os.path.exists(file_path):
            logging.error(f"文件不存在: {file_path}")  # 明确提示缺失的文件
            return False
    return True
# ============ 运行入口 ============
if __name__ == "__main__":
    # 初始化API客户端
    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    # 检查检索结果文件是否存在
    if not check_retrieval_files_exist(Config.RETRIEVAL_RESULTS_PATH):
        logging.info("检索结果文件不存在，开始执行检索...")

        # 确保模型保存路径存在
        os.makedirs(Config.model_save_path, exist_ok=True)

        # 初始化模型组件
        model = AutoModel.from_pretrained(r"D:\model临时\BAAI\bge-large-zh-v1___5").to(Config.DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(r"D:\model临时\BAAI\bge-large-zh-v1___5")
        model.eval()

        # 初始化稀疏编码器
        vectorizer = CountVectorizer()
        df = pd.read_csv(Config.CSV_PATH)
        vectorizer.fit(df['imagefinding'].tolist())

        # 初始化Milvus连接（仅在需要检索时执行）
        try:
            connections.connect("default", host=Config.MILVUS_HOST, port=Config.MILVUS_PORT, timeout=30)  # 增加超时参数
            collection = setup_collection()  # 集合初始化必须在连接之后
        except Exception as e:
            logging.error(f"Milvus连接失败: {str(e)}")
            raise SystemExit("程序因数据库连接失败终止")

        # 执行检索
        retrieve_all_queries(Config.CSV_PATH, Config.RETRIEVAL_RESULTS_PATH)

        # 主动释放资源
        connections.disconnect("default")
        del model, tokenizer, collection  # 显式释放GPU资源
        torch.cuda.empty_cache()  # 清空GPU缓存
    else:
        logging.info("检索结果文件已存在，跳过所有检索相关操作...")

    # API处理阶段（完全独立于Milvus）
    process_results_with_api(Config.RETRIEVAL_RESULTS_PATH, Config.OUTPUT_CSV)