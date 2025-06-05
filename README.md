# Research Paper Analysis & Classification Pipeline

## Project Overview
The goal of this project is to develop a model that automates labeling of the text in the "Text" column of a given dataset. The dataset contains two different labels/classes, and the objective is to accurately classify opinions related to Research Paper Analysis & Classification based on these categories.

## Approach
To tackle this problem, two distinct techniques were employed:

### 1. Mistral-OpenOrca (LLM) Model
- Utilized the Mistral-OpenOrca model hosted locally on the Ollama server.
- Created a prompt specifying the prediction of Disease-Specific Identifications for different classes, enabling the LLM to label the text accordingly.
- No model training was performed in this approach.


### 2. DistilBERT-base-uncased Model
- Employed the DistilBERT-base-uncased model for task-specific fine-tuning.
- Selected an equal number of samples from 2 different classes to create a balanced dataset.
- Evaluated the performance of the fine-tuned model using a lightweight model due to computational limitations and a very limited number of data points.

## Requirements & Installation

### Installing Ollama
- For detailed installation instructions, please visit: [Ollama Installation](https://ollama.com/download/windows).

### Pulling the Mistral-OpenOrca Model
- After installation, you can pull the Mistral-OpenOrca model by running the following command in your terminal:
ollama pull mistral-openorca:latest


## Dependencies
A requirements.txt file is included with the necessary dependencies for this project. Ensure to install all required packages listed in this file.

## Observations

Comments have been added in the code to enhance understanding.

## Conclusion
Based on the exercise, we can clearly observe that fine-tuning enhanced the model’s performance in terms of accuracy and F1 score, with improvements of 60% and 50%, respectively. Fortunately, we had a balanced dataset. Otherwise, we could have modified the loss function to assign higher weights to the minority class, thereby penalizing its misclassification more heavily.

Hyperparameter tuning can also significantly improve model performance. During the fine-tuning phase, we recommend utilizing the LoRA (Low-Rank Adaptation) approach, as it is particularly effective for parameter-efficient training, requiring fewer computational resources. It's important to experiment with different hyperparameters to find the optimal configuration for the model. Key LoRA hyperparameters—such as r, target modules (I used q_lin), and alpha—play a crucial role in fine-tuning effectiveness.

We must also carefully monitor for overfitting. If the model shows signs of overfitting, we can either add more training data (though this can be expensive) or reduce the value of r (in my case, I used r = 4 to lower computational costs). Additional techniques such as increasing dropout or raising the weight decay rate in optimizers like AdamW or SGD can also help mitigate overfitting.

Another important hyperparameter is alpha, which influences the contribution of the adapter layers. It is generally recommended to set alpha to 1–2 times the value of r.

Evaluating the model's performance on unseen data is essential to ensure it generalizes well beyond the training examples.

We can also consider implementing QLoRA, a quantized version of LoRA. While QLoRA may result in some performance degradation, it offers approximately 33% savings in GPU memory usage. However, this comes at the cost of a 39% increase in training time due to the additional quantization and dequantization steps applied to the pretrained model weights.

Fine-tuning models like BERT requires thoughtful hyperparameter optimization. Effective tuning strategies include grid search for smaller parameter sets, random search for faster exploration, and Optuna for intelligent, performance-driven optimization.

I highly recommend testing the model’s performance with different hyperparameter settings, experimenting with various base models, and evaluating performance in a cost-effective manner.

Deployment Strategy 
Building a machine learning model is always incomplete unless it is deployed and tested with real-world data. In the section below, I discuss some aspects of model deployment.
When it comes to deployment, I like FastAPI.  
1. FastAPI 
FastAPI handles requests asynchronously and offers the following advantages:
FastAPI efficiently manages API endpoints with automatic data validation and asynchronous request handling.
TorchServe serves the model efficiently, enabling scalable and production-ready deployment for PyTorch models.

3. Hugging Face Inference API
For quick and easy deployment, the Hugging Face Inference API is a lifesaver. 
Upload your model to the Hugging Face Model Hub.
Use their API to call the model.


4. Real-World Considerations
Latency: FastAPI with TorchServe offers faster performance for real-time predictions.
Ease of Use: Hugging Face’s API is great for quick deployments but can become expensive with high traffic.
Scalability: Containerizing your model with Docker and deploying it on Kubernetes is ideal for large-scale applications.
Virtual Layerwise Learning Memory(VLLM): It is a high-performance, optimized inference and serving framework specifically designed for Large Language Models (LLMs). Developed to maximize throughput and minimize latency during inference, VLLM enables the efficient serving of large-scale models like GPT, OPT, LLaMA, and others. I've never used it.
To optimize costs, it's important to profile the model’s inference time and batch requests whenever possible [1].
Overall, this project is feasible and achievable with careful planning and execution. By following these steps, we can enhance the model's effectiveness and ensure it meets our business goals.

Link: https://medium.com/@heyamit10/fine-tuning-bert-for-sentiment-analysis-a-practical-guide-f3d9c9cac236 

Link: https://www.philschmid.de/fine-tune-llms-in-2024-with-trl 


