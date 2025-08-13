## 💎 Gemma Garage - Democratizing LLM Fine-tuning

[Gemma Garage](https://gemma-garage.web.app/home) is a web platform designed to make it easy to fine-tune Gemma models, Google DeepMind's family of open-weights language models. 

It allows for intelligent ingestion, parsing, and augmentation of files in a wide variaty of formats (PDF, JSON, TXT, etc), and abstracts away the complexity

of fine-tuning from the user.

With Gemma Garage,

you can:

✅  Create your own account, where your projects will be stored

✅ Ingest structured or unstructured data (e.g. PDF's) and have it augmented. For instance, if you have a limited dataset
and would like to train a chatbot on it, the augmentation features are perfect for you. Alternatively, you can use datasets from Hugging Face.

✅ Choose different models and hyperparameters.

✅ Push your model to Hugging Face with just a few clicks. Gemma Garage supports Hugging Face integration with OAuth, so in a matter of seconds you can 
log in to your HF account and push your model.

✅  Test the finetuned model against the base one.

To get started, go to https://gemma-garage.web.app/home.


### How does it work?

Gemma Garage is a comprehensive suite for building, finetuning, and deploying Large Language Models (LLMs), with a focus on Gemma models. This project leverages Google Cloud services, efficient finetuning techniques with Unsloth, and a structured approach to data augmentation.

## Architecture Overview

The Gemma Garage project is composed of several interconnected services, each handling a specific part of the LLM lifecycle:

- **Frontend:** A React-based web application providing a user interface for interacting with the LLM services.
- **Backend:** A FastAPI service acting as the central API gateway, orchestrating requests between the frontend and other specialized backend services.
- **Augmentation Service:** A FastAPI service responsible for generating high-quality synthetic datasets for LLM finetuning, utilizing the Synthetic Data Kit.
- **Finetuning Service:** A FastAPI service dedicated to finetuning LLMs, leveraging Unsloth for efficiency and integrating with Google Cloud AI Platform.
- **Inference Service:** A FastAPI service for serving finetuned LLMs for inference, optimized for performance.

## Components

### 1. `gemma-garage-frontend`

- **Description:** The user-facing web application built with React. It provides an intuitive interface for users to interact with the Gemma Garage functionalities, such as initiating finetuning jobs, managing data, and performing inference.
- **Key Technologies:**
    - **React:** A JavaScript library for building user interfaces.
    - **Material UI (`@mui/material`):** A popular React UI framework implementing Google's Material Design.
    - **Firebase:** Used for authentication, real-time database, and other backend services (e.g., user management, data storage).
    - **React Router DOM:** For declarative routing in the React application.
    - **Chart.js & React Chart.js 2:** For data visualization and displaying metrics.

### 2. `gemma-garage-backend`

- **Description:** The core API gateway for the Gemma Garage. It handles user requests from the frontend, manages data flow, and communicates with other specialized backend services.
- **Key Technologies:**
    - **FastAPI:** A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.
    - **Uvicorn:** An ASGI server for running FastAPI applications.
    - **Google Cloud Integration:** Extensive use of Google Cloud services for various functionalities:
        - **Google Cloud AI Platform:** For managing machine learning workflows and potentially deploying models.
        - **Google Cloud Firestore:** A NoSQL document database for storing application data.
        - **Google Cloud Storage:** For storing large files, datasets, and model artifacts.
        - **Google Generative AI:** For interacting with Google's own generative AI models.
        - **Google Cloud Logging:** For centralized logging and monitoring.
    - **Hugging Face Libraries (`huggingface-hub`, `datasets`):** For interacting with Hugging Face models and datasets.
    - **Pandas & NumPy:** For data manipulation and analysis.

### 3. `gemma-garage-augmentation`

- **Description:** This service is responsible for generating synthetic data, which is crucial for expanding and diversifying training datasets for LLMs. It leverages the `synthetic-data-kit` to create high-quality reasoning traces and QA pairs.
- **Key Technologies:**
    - **FastAPI & Uvicorn:** Provides an API interface for data augmentation tasks.
    - **Google Cloud Storage:** For storing and retrieving augmented datasets.
    - **Synthetic Data Kit:** A specialized tool (likely integrated as a library or a separate process) for:
        - **Ingesting** various data formats (PDF, HTML, YouTube, DOCX, PPTX, TXT).
        - **Creating** synthetic data (QA pairs, Chain-of-Thought reasoning).
        - **Curating** generated data based on quality.
        - **Saving** data in various finetuning-friendly formats.
        - Can utilize a vLLM server or external API endpoint (potentially the `gemma-garage-inference` service) as its LLM backend for data generation.

### 4. `gemma-garage-finetuning`

- **Description:** This service orchestrates the finetuning of Large Language Models, with a particular emphasis on efficiency and resource optimization. It utilizes Unsloth to accelerate the finetuning process.
- **Key Technologies:**
    - **FastAPI & Uvicorn:** Provides an API for initiating and managing finetuning jobs.
    - **Unsloth:** A powerful library that enables 2x faster LLM finetuning with up to 80% less VRAM. It achieves this through optimized kernels and efficient memory management, making it ideal for finetuning Gemma and other LLMs on GPUs.
    - **Hugging Face Ecosystem (`transformers`, `peft`, `accelerate`, `trl`):** Core libraries for building, finetuning, and deploying transformer-based models.
        - **PEFT (Parameter-Efficient Finetuning):** Techniques like LoRA for efficient adaptation of large models.
        - **Accelerate:** Simplifies distributed training and mixed-precision training.
        - **TRL (Transformer Reinforcement Learning):** For advanced finetuning techniques like DPO (Direct Preference Optimization).
    - **BitsAndBytes:** For 8-bit and 4-bit quantization, further reducing memory footprint during finetuning.
    - **PyTorch:** The underlying deep learning framework.
    - **Google Cloud AI Platform:** For managing and executing finetuning jobs, often leveraging GPU-enabled instances.
    - **Google Cloud Storage & `gcsfs`:** For storing finetuning datasets, checkpoints, and finetuned models.
    - **LiteLLM:** A library for simplifying interactions with various LLM APIs.
    - **Google Generative AI:** For potential integration with Google's own generative AI services during finetuning or evaluation.
    - **TensorBoard:** For visualizing training metrics and monitoring finetuning progress.

### 5. `gemma-garage-inference`

- **Description:** This service is responsible for serving finetuned LLMs for real-time inference. It's optimized for performance and can leverage GPU resources.
- **Key Technologies:**
    - **FastAPI & Uvicorn:** Provides an API endpoint for model inference.
    - **PyTorch:** The deep learning framework for running models.
    - **Hugging Face Ecosystem (`transformers`, `peft`, `bitsandbytes`, `huggingface_hub`):** For loading and running LLMs, including those finetuned with PEFT and quantized models.
    - **Google Cloud AI Platform:** Likely used for deploying and managing the inference endpoints, especially when using GPU-accelerated instances.
    - **Google Generative AI:** For potentially serving Google's own generative AI models or integrating with them.

## Deployment and Infrastructure

The project is designed to be deployed on **Google Cloud Run with GPU support**. Each service (`backend`, `augmentation`, `finetuning`, `inference`) is likely containerized (Docker) and deployed as a separate Cloud Run service.

- **Google Cloud Run:** A fully managed compute platform for deploying containerized applications. Its ability to scale automatically and support GPUs makes it suitable for LLM workloads.
- **Google Cloud Build:** Used for continuous integration and deployment (CI/CD), as indicated by `cloudbuild.yaml` files in various service directories. This automates the process of building Docker images and deploying them to Cloud Run.
- **Firebase:** As seen in the frontend, Firebase is used for user management and potentially other client-side backend needs.

## How it Works (End-to-End Flow)

1.  **User Interaction (Frontend):** Users interact with the React frontend to initiate tasks like data augmentation or LLM finetuning.
2.  **API Gateway (Backend):** The frontend communicates with the `gemma-garage-backend` service, which acts as an API gateway.
3.  **Data Augmentation (Augmentation Service):** If a user requests data augmentation, the backend triggers the `gemma-garage-augmentation` service. This service uses the `synthetic-data-kit` to generate new datasets, potentially using the `gemma-garage-inference` service as an LLM backend for data generation. The augmented data is stored in Google Cloud Storage.
4.  **LLM Finetuning (Finetuning Service):** When a finetuning job is initiated, the `gemma-garage-finetuning` service takes over. It retrieves data from Google Cloud Storage, uses Unsloth and other Hugging Face libraries to efficiently finetune the specified LLM on GPU-enabled instances (likely via Google Cloud AI Platform), and saves the finetuned model back to Google Cloud Storage.
5.  **LLM Inference (Inference Service):** For real-time predictions or text generation, the `gemma-garage-inference` service loads the desired LLM (either a pre-trained model or a finetuned one from GCS) and serves inference requests. This service is optimized for low-latency responses.
6.  **Monitoring and Logging:** All services integrate with Google Cloud Logging for centralized log collection and monitoring.

## Getting Started

-   **Prerequisites:** Google Cloud Project, Firebase Project, Python, Node.js, Docker.
-   **Local Setup:** Instructions for running each service locally.
-   **Deployment:** Instructions for deploying to Google Cloud Run.





