# Automatic Question Generation for Educational Content

## Overview

This project focuses on automatic question generation using Transformer-based models, specifically targeting educational content to create meaningful and relevant questions that can help learners test their understanding. The project utilizes T5 (Text-To-Text Transfer Transformer) for generating questions and employs advanced metrics like BLEU scores and cosine similarity to evaluate the quality of the generated questions. Additionally, the generated questions are analyzed using visualizations to provide better insights into their quality.

## Project Structure

- **Technologies Used**: Transformer (T5), LSTM, PyTorch, Transformers Library, SentenceTransformer, NLTK, Matplotlib, Seaborn.
- **Code Sections**:
  1. Loading the T5 Transformer model for question generation.
  2. Pre-trained SentenceTransformer for semantic similarity evaluation.
  3. Calculation of BLEU scores for evaluating linguistic similarity.
  4. Cosine similarity analysis for semantic relevance.
  5. Visualization for easy interpretation of results.

## Objective

The main objective of this project is to automate the generation of educational questions, making learning more interactive and engaging. We aim to generate high-quality questions that accurately reflect the information in the given text, ensuring that learners can use them effectively to self-assess their understanding.

## Methodology

### 1. Question Generation

We used the T5 Transformer model (“t5-large” variant) to generate questions based on a given text passage. The T5 model is known for its ability to generate coherent, context-relevant content by treating every NLP problem as a text-to-text problem. In this case, we used T5 to transform an educational passage into questions that a student might ask to test their understanding.

For this project, a passage on the programming language Python was used as a demonstration. The T5 model was used to generate three questions based on the provided passage.

### 2. Evaluation Metrics

To assess the quality of the generated questions, we employed two main metrics: BLEU Score and Cosine Similarity Score.

#### **BLEU Score (Bilingual Evaluation Understudy)**

BLEU score is a metric commonly used for evaluating the quality of text that has been machine-translated from one language to another. In this context, BLEU is used to compare the generated questions with reference questions to evaluate how similar they are linguistically. The higher the BLEU score, the closer the generated text is to the reference.

- **BLEU Score Calculation**: The BLEU score is calculated for each generated question against a set of reference questions. We use a smoothing function to avoid zero scores in cases where there is no n-gram overlap.
- **Results**: The BLEU scores of the generated questions were **0.16**, **0.64**, and **0.07**. The moderate score of **0.64** indicates some linguistic overlap, while lower scores indicate less similarity in wording compared to reference questions.

#### **Cosine Similarity Score**

Cosine similarity is used to measure the semantic similarity between the generated and reference questions. This metric calculates the cosine of the angle between two non-zero vectors, which, in this context, represent the encoded representations of the questions. A higher cosine similarity score indicates a greater degree of semantic overlap.

- **Cosine Similarity Calculation**: Using the pre-trained SentenceTransformer model (“all-MiniLM-L6-v2”), we calculated cosine similarity scores between the generated questions and reference questions to determine how semantically similar they are.
- **Results**: The cosine similarity scores for the generated questions were **0.94**, **0.99**, and **0.97**, indicating high semantic similarity. This suggests that while the generated questions may not always have the same wording as the reference questions (hence the lower BLEU scores), they still convey similar meanings effectively.

### 3. Visualizations

To provide better insight into the evaluation results, we created visualizations for both BLEU scores and cosine similarity scores using Matplotlib and Seaborn.

- **BLEU Score Distribution**: A bar plot was generated to show the BLEU scores of the generated questions. This helps in understanding the linguistic overlap between the generated and reference questions.

  ![image](https://github.com/user-attachments/assets/46025060-d96b-4240-bb17-e17cd01be88d)

- **Cosine Similarity Distribution**: Another bar plot was generated to show the cosine similarity scores, providing an understanding of the semantic overlap between the generated and reference questions.

![image](https://github.com/user-attachments/assets/30470137-c8da-40df-b9c2-5e423b60e8a2)


These visualizations are critical in helping us assess the effectiveness of our model in generating meaningful and accurate questions.

## Summary of Results

- **Generated Questions**: The questions generated by the model were generally relevant to the passage but often lacked sufficient linguistic overlap with the reference questions, as indicated by the varying BLEU scores.
- **Semantic Relevance**: The cosine similarity scores indicated that the questions had high semantic similarity with the reference questions, even when linguistic similarity was low.

The combination of low to moderate BLEU scores and high cosine similarity scores indicates that the generated questions, while semantically relevant, may benefit from more precise wording or additional training to improve fluency and coherence.

## Key Insights and Future Improvements

- **Improving Linguistic Fluency**: The generated questions often had lower BLEU scores, suggesting that the linguistic structure could be refined. Fine-tuning the T5 model with a dataset specific to question generation in educational contexts might help improve fluency and linguistic quality.
- **Enhancing Semantic Relevance**: Despite high cosine similarity, further improvements could involve incorporating more diverse training data and adjusting model parameters to better understand context.
- **Interactive Question Generation**: Adding a mechanism for human feedback during training could also enhance the model's ability to generate high-quality, contextually appropriate questions.

## How to Run the Project

1. **Clone the Repository**:
   ```
   git clone <repository_link>
   ```
2. **Install Dependencies**: Install the necessary dependencies using the `requirements.txt` file.
   ```
   pip install -r requirements.txt
   ```
3. **Run the Script**: Execute the script to generate questions and visualize evaluation metrics.
   ```
   python question_generation.py
   ```

## Dependencies

- Python 3.7+
- Transformers (Hugging Face)
- PyTorch
- SentenceTransformer
- Matplotlib
- Seaborn
- NLTK

## Conclusion

This project demonstrates the potential of using transformer models for automatic question generation, with a focus on generating educational questions that help learners test their understanding. While the generated questions were semantically relevant, there is room for improvement in terms of linguistic structure and fluency. Future iterations of this project will focus on improving both the quality and diversity of the generated questions to create a more effective educational tool.

## Contact

If you have any questions or suggestions regarding this project, feel free to reach out:

- **Pushpendra Singh**
- [Email](mailto:spushpendra540@gmail.com)
- [GitHub](https://github.com/sirisaacnewton540)
