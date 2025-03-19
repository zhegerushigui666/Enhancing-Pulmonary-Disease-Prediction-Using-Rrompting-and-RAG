---

# Enhancing Pulmonary Disease Prediction Using Prompting and RAG

## Project Overview
This repository contains the implementation code for the paper titled **"Enhancing Pulmonary Disease Prediction Using Large Language Models with Feature Summarization and Hybrid Retrieval-Augmented Generation"**. The paper proposes a novel approach combining feature summarization and hybrid Retrieval-Augmented Generation (RAG) to improve the accuracy of pulmonary disease prediction. This codebase provides the implementation of the algorithms, experimental scripts, and data analysis tools.

## Paper Information
• **Title**: Enhancing Pulmonary Disease Prediction Using Large Language Models with Feature Summarization and Hybrid Retrieval-Augmented Generation  
• **Authors**: Ronghao Li<sup>1,2</sup>#, Shuai Mao<sup>3</sup>#, Congmin Zhu<sup>1,2</sup>, Yingliang Yang<sup>1,2</sup>, Chunting Tan<sup>4</sup>, Li Li<sup>5</sup>, Xiangdong Mu<sup>5</sup>, Honglei Liu<sup>1,2*</sup>, Yuqing Yang<sup>3*</sup>  
• **Journal**: JMIR (Journal of Medical Internet Research)  
• **Paper Link**: [Link to Paper](#)  

## Code Structure
Below is a description of the main files in the repository and their functionalities:

### 1. `new_retriever.py`
• **Functionality**: Used for running SHAP analysis.  
• **Usage**: If you need to customize the retrieval mode for Milvus, you can directly call the functions in this file.  
• **Note**: By default, all requests in `delivery.py` use a dual-path recall strategy.

### 2. `delivery.py`
• **Functionality**: Calls the API and tests using pre-edited prompts, saving the results to a specified CSV file.

### 3. `milvus_build.py`
• **Functionality**: Uses an embedding model to convert data from a specified CSV file into vectors and stores them in the Milvus vector database.


## Installation and Usage

### Environment Requirements
• Python 3.x  
• Dependencies:  
  • **torch**  
  • **pandas**  
  • **transformers** (for `AutoTokenizer` and `AutoModel`)  
  • **pathlib**  
  • **pymilvus** (for Milvus vector database operations)  
  • **scikit-learn** (for `CountVectorizer`)  
  • **openai** (for OpenAI API integration)  
  • **zhipuai** (for ZhipuAI API integration)  
  • **logging**, **os**, **json**, **time** (standard Python libraries)  


This version lists the libraries in a more readable format. Let me know if you'd like further adjustments!

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/zhegerushigui666/Enhancing-Pulmonary-Disease-Prediction-Using-Rrompting-and-RAG.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code
• To run SHAP analysis:
  ```bash
  python new_retriever.py --your-args
  ```
• To test prompts and save results:
  ```bash
  python delivery.py
  ```
• To build the Milvus vector database:
  ```bash
  python milvus_build.py
  ```

## Milvus Installation
We use **Milvus v2.4.1**. For installation instructions, refer to the official Milvus documentation:  
[Milvus Installation Guide](https://github.com/milvus-io/milvus)

## Contributing
We welcome contributions! Please open an issue or submit a pull request. Ensure your code follows the existing style and passes all tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
We would like to thank [relevant institutions/individuals] for their support and assistance.


---