

# **Parallel Computing Impact on ML Models: A Comparative Analysis** 🚀

This project provides a comprehensive comparative analysis of how parallel computing affects the performance of various machine learning models. We systematically benchmark a range of algorithms, from classical models like Logistic Regression and Random Forest to advanced gradient boosting frameworks like XGBoost and CatBoost. Additionally, we conduct a deep dive into the performance of a neural network, comparing its execution on a multi-core CPU against an NVIDIA Tesla T4 GPU.

Our analysis focuses on key metrics such as training time, speedup, accuracy, and efficiency to provide data-driven insights for optimizing machine learning workflows on different hardware configurations.

### **Team Members**

  - **Hussain Ahmad** (22i-1374)
  - **Asad Ullah** (22I-0589)
  - **Sami Naeem** (22I-0587)

-----

## **Project Overview**

The core objective of this project is to demystify the relationship between machine learning algorithms and parallel processing capabilities. We investigate how different models scale with an increasing number of CPU cores and how a deep neural network benefits from GPU acceleration.

Our methodology involves:

1.  **Systematic Benchmarking:** Running each model with a varying number of cores/threads.
2.  **Performance Metrics:** Recording training time, accuracy, and F1 score for each configuration.
3.  **Comparative Analysis:** Contrasting the results across different models to understand their unique parallelization characteristics.
4.  **CPU vs. GPU:** Quantifying the performance gain of a neural network on a Tesla T4 GPU compared to a local CPU.

This work serves as a practical guide for data scientists and engineers, helping them make informed decisions about resource allocation and algorithm selection for their specific computational environments.

-----

## **Detailed Model Analysis**

We have analyzed the parallelization efficiency of several machine learning models. Each section below summarizes the key findings for the respective algorithm.

### **Model 1: Random Forest**

**Parallelization Strategy:** Random Forest is an **embarrassingly parallel** algorithm. The training of each individual decision tree within the ensemble is an independent task, making it highly suitable for parallelization via `scikit-learn`'s `n_jobs` parameter.
**Key Findings:**

  - **Significant Speedup:** A near-linear speedup is observed initially as the core count increases.
  - **Diminishing Returns:** The speedup eventually plateaus due to overhead from inter-process communication and the fact that some tasks (like data splitting) remain serial.
  - **Optimal Configuration:** The analysis helps identify the optimal number of cores that provides the best trade-off between speedup and resource utilization for a given number of estimators.

### **Model 2: Decision Tree**

**Parallelization Strategy:** A single Decision Tree classifier in `scikit-learn` does **not support parallelization** during training. Its recursive splitting process is inherently sequential.
**Key Findings:**

  - **Consistent Performance:** As expected, training time remains constant regardless of the number of cores specified.
  - **Methodological Control:** This model served as a crucial control group, highlighting the fundamental difference between algorithms that are inherently parallelizable and those that are not.

### **Model 3: Logistic Regression**

**Parallelization Strategy:** The parallelization capability of Logistic Regression depends on the chosen solver. Our experiment used the `lbfgs` solver, which can utilize `n_jobs` to parallelize certain computations.
**Key Findings:**

  - **Amdahl's Law:** The observed speedup closely follows Amdahl's Law, demonstrating a theoretical limit on the acceleration that can be achieved.
  - **Limited Parallelism:** Since only parts of the optimization process can be parallelized, the speedup is modest compared to ensemble methods. The performance gain diminishes rapidly after a few cores due to communication overhead.

### **Model 4: XGBoost**

**Parallelization Strategy:** XGBoost is a highly optimized gradient boosting library with built-in parallelism at multiple levels (data, feature, tree). It effectively leverages multi-core CPUs via its `n_jobs` parameter.
**Key Findings:**

  - **High Efficiency:** XGBoost is designed for high-performance computing and shows excellent scaling with an increasing number of cores.
  - **Impact of Tree Depth:** The benefits of parallelization are more pronounced with **deeper trees** (`max_depth > 6`). Deeper trees involve more complex computations (finding the best split), which are highly parallelizable, leading to a more significant speedup.

### **Model 5: Gaussian Naive Bayes**

**Parallelization Strategy:** Similar to the single Decision Tree, the `scikit-learn` implementation of Gaussian Naive Bayes does **not support parallelization** during training.
**Key Findings:**

  - **Inherent Serial Nature:** This experiment confirmed the algorithm's serial nature. Any minor fluctuations in training time across different core counts are attributed to system-level noise and background processes, not algorithmic speedup.
  - **Baseline for Comparison:** This model provides a vital baseline for understanding the inherent variability of performance measurements.

### **Model 6: K-Nearest Neighbors (KNN)**

**Parallelization Strategy:** KNN is a lazy learning algorithm. The "training" phase is trivial (just storing the data). Parallelization primarily accelerates the **prediction phase**, which involves calculating distances from each test point to all training points.
**Key Findings:**

  - **Prediction-Phase Speedup:** The most significant performance gains are observed during the prediction phase, as distance calculations are an embarrassingly parallel task.
  - **Scalability:** The efficiency of parallelization is directly related to the size and dimensionality of the dataset. Larger datasets yield a greater benefit from multi-core processing during prediction.

### **Model 7: CatBoost**

**Parallelization Strategy:** CatBoost is a gradient boosting library optimized for parallel tree construction and implements parallelism at the data, feature, and node levels using `thread_count`.
**Key Findings:**

  - **Designed for Parallelism:** CatBoost is engineered for efficiency and demonstrates excellent scaling.
  - **Clear Speedup:** A clear and consistent speedup is observed as the number of threads increases, confirming its ability to effectively utilize modern multi-core processors.

-----

## **Neural Network Analysis: CPU vs. GPU Acceleration**

This section presents a detailed comparison of a binary classification neural network pipeline executed on two different hardware platforms: a multi-core CPU and an NVIDIA Tesla T4 GPU.

### **GPU Implementation (Tesla T4)**

This pipeline is optimized for GPU acceleration, leveraging the Tesla T4's architecture. Key optimizations include:

  - **Mixed Precision Training:** Using PyTorch's Automatic Mixed Precision (AMP) with `autocast` and `GradScaler` to perform computations in FP16 (half-precision) for a significant speed boost, while maintaining model stability.
  - **Memory Optimization:** Using `pin_memory=True` and `non_blocking=True` for efficient data transfers from CPU to GPU, which is crucial for maximizing the GPU's utilization.
  - **Large Batch Size:** A batch size of 8192 is used to take full advantage of the GPU's massive parallelism and memory bandwidth.

### **CPU Implementation (Intel i7-3770)**

This implementation runs the identical neural network pipeline on a CPU. All GPU-specific optimizations, such as mixed precision and CUDA synchronizations, are disabled. This provides a direct, fair comparison of raw performance.

### **Key Findings: The Performance Gap**

The analysis reveals a significant **performance gap** between the CPU and GPU implementations.

  - **Computation Intensity:** The deep network architecture (6 hidden layers with wide layers) involves heavy matrix multiplications, a task for which GPUs with thousands of CUDA cores are perfectly suited.
  - **Hardware Limitations:** The Intel i7-3770 (4 cores, 8 threads) is limited in its ability to parallelize these operations, leading to significantly longer training times. The GPU's superior memory bandwidth and specialized architecture for parallel computation create an overwhelming advantage.
  - **Speedup:** The GPU-accelerated pipeline demonstrates a massive speedup, confirming that for deep learning workloads, specialized hardware is not just a convenience but a necessity for achieving reasonable training times.

-----

## **How to Run the Code**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/parallel-computing-ml.git
    cd parallel-computing-ml
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Explore the notebooks:**
    Launch Jupyter (via `jupyter notebook` or `jupyter lab`) and open the `.ipynb` files in the `notebooks/` directory to run the experiments and view the results.

-----

## **Repository Structure**

  - `data/`: Contains the dataset used for the experiments.
  - `notebooks/`: All Jupyter notebooks for each model's analysis.
  - `results/`: Directory for storing generated plots and dataframes.
  - `report/`: The final project report in Word format.
  - `README.md`: This file.
  - `requirements.txt`: Python dependencies.

-----

## **License**

This project is licensed under the **MIT License**.

