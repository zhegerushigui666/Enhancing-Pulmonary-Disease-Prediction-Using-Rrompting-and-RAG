import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
# 读取CSV文件
file_path = r""# 替换为你的文件路径

#out_path = fr"C:\Users\Lenovo\Desktop\self-adaptive\gpto4长庚验证友谊1output_v{time.strftime('%Y%m%d%H%M')}.csv"
data = pd.read_csv(file_path)

# 每四行选取一行（除去表头）
#data = data.iloc[::4, :].reset_index(drop=True)

# 按照 'Finding' 列去重，保留第一个出现的记录
#data = data.drop_duplicates(subset='Finding', keep='first')

# 提取实际标签
true_labels = data['Real Label']


# 定义一个函数来提取特定的疾病名称或者'无病'
def extract_first_disease_label(label_string):
    # 正则表达式模式，用于查找指定的疾病名称
    pattern = r'(?:肺炎|肺癌|肺结核|无病)'
    # 获取最后一个冒号之后的部分，并去除首尾空白
    last_part = label_string.replace("：",":").split(':')[-1]
    # 使用re.findall找到所有匹配项
    matches = re.findall(pattern, last_part)
    # 如果找到任何匹配项，则返回第一个匹配项，否则返回'无病'
    return matches[0] if matches else '无病'

# 应用这个函数到data的'Predicted Label'列
predicted_labels = data['Predicted Label'].apply(extract_first_disease_label)

# 从预测列提取最后一个冒号后的内容作为预测结果
#predicted_labels = data['Predicted Label'].apply(lambda x: x.split(':')[-1].strip())
#predicted_labels = data['Predicted Label']
#sparse_predicted_label
#dense_predicted_label
#hybrid_predicted_label


# 根据关键词更新预测标签
def update_label(label):
    if '肺炎' in label:
        return '肺炎'
    elif '肺结核' in label:
        return '肺结核'
    elif '无病' in label:
        return '无病'
    elif '肺癌' in label:
        return '肺癌'
    else:
        return label  # 如果没有匹配到，保留原标签

# 新增列 p_labels 存储更新后的预测标签
data['p_labels'] = predicted_labels.apply(update_label)

# 保存到源文件（覆盖或另存为新文件）
#data.to_csv(out_path, index=False, encoding="utf-8-sig")
# 如果需要另存为新文件，可以指定新路径，如 'data_updated.csv'

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# 定义中文标签和对应的英文标签
labels = ['肺结核', '肺炎', '肺癌', '无病']
english_labels = ['TB', 'PN', 'LC', 'ND']

# 确保更新后的标签符合定义的标签
assert set(data['p_labels']).issubset(labels), "更新后的预测标签存在未定义的类别，请检查代码逻辑！"

# 计算分类报告
report = classification_report(
    true_labels,
    data['p_labels'],
    labels=labels,
    target_names=english_labels,
    output_dict=True,
    zero_division=0  # 如果除以零，直接返回0
)

# 计算混淆矩阵
conf_matrix = confusion_matrix(true_labels, data['p_labels'], labels=labels)

# 计算 accuracy、precision、recall 和 f1-score
accuracy = accuracy_score(true_labels, data['p_labels'])
# 计算宏观精度、召回率和F1分数
# 这里同样添加 zero_division 参数
macro_precision = precision_score(true_labels, data['p_labels'], average='macro', zero_division=0)
macro_recall = recall_score(true_labels, data['p_labels'], average='macro', zero_division=0)
macro_f1 = f1_score(true_labels, data['p_labels'], average='macro', zero_division=0)

# 输出结果
print(f'Accuracy: {accuracy:.4f}')
print(f'Macro Precision: {macro_precision:.4f}')
print(f'Macro Recall: {macro_recall:.4f}')
print(f'Macro F1-Score: {macro_f1:.4f}')

# 创建一个DataFrame以便于绘图
conf_matrix_df = pd.DataFrame(conf_matrix, index=english_labels, columns=english_labels)

# 设置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置绘图大小
plt.figure(figsize=(10, 7))

# 使用seaborn绘制热力图
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=True,  # 去掉颜色条
            xticklabels=english_labels, yticklabels=english_labels,
            annot_kws={'size': 36})  # 调整注释文本的大小

# 设置坐标轴标签的字体大小
plt.xticks(fontsize=36)  # 调整X轴标签的字体大小
plt.yticks(fontsize=36)  # 调整Y轴标签的字体大小

# 去掉图标题
# plt.title('Confusion Matrix', fontsize=18)  # 注释掉或删除

# 设置X轴和Y轴标签
plt.xlabel('Predicted Label', fontsize=24)  # 调整X轴标签的字体大小
plt.ylabel('True Label', fontsize=24)  # 调整Y轴标签的字体大小



# 设定文件保存路径和文件名
#save_path = rf"C:\Users\Lenovo\Desktop\self-adaptive\字号确定36 deepseek202438622混淆矩阵-复现-FS-COT-RAG-Sparse-time{time.strftime('%Y%m%d%H%M')}.png"  # 手动设置文件名

# 保存绘制好的混淆矩阵图像
#plt.savefig(save_path, format='png', bbox_inches='tight')  # bbox_inches='tight'可以避免图像边缘被裁剪

# 展示图形
plt.show()



