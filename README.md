# T3SS: The Tier-3 Search System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Code Coverage](https://img.shields.io/badge/coverage-85%25-blue)
![License](https://img.shields.io/badge/license-Proprietary-red)

T3SS is a next-generation, scalable, and modular web search engine designed to handle the entire lifecycle of web data—from discovery and indexing to intelligent querying and ranking. It's built on a polyglot microservices architecture, leveraging the best language for each specific task.

***

## ✨ Core Features

* **Massive-Scale Web Crawling**: A distributed, polite, and efficient crawling system capable of discovering and fetching content from the web.
* **Advanced Indexing Pipeline**: Features an inverted index, a document store, and a link graph builder with deduplication and spam detection.
* **Multi-Factor Ranking**: Combines classic signals like PageRank with modern machine learning models and contextual signals for highly relevant results.
* **Semantic & Vector Search**: A deep NLP core provides semantic understanding, embedding generation, and vector search capabilities.
* **Vertical Search Engines**: Specialized, fine-tuned search experiences for Images, News, Maps, Scholarly articles, and more.
* **Cloud-Native & Scalable**: Designed to be deployed on modern cloud infrastructure using Docker, Kubernetes, and Terraform for ultimate scalability and resilience.

***

## 🏛️ Architecture Overview

The T3SS project is built on a **polyglot microservices architecture**. This means that different components are developed as independent services, often using different programming languages chosen for their strengths in a particular domain.

The project is organized into these primary top-level directories:

* **`/core`**: Contains the heart of the search engine, including all logic for crawling, indexing, and querying.
* **`/frontend`**: Holds all user-facing components, including the web UI, mobile handlers, and API endpoints.
* **`/backend_services`**: A collection of supporting microservices for tasks like authentication, ads, billing, and machine learning pipelines.
* **`/infrastructure`**: All configuration and code related to deployment, monitoring, CI/CD, and testing.
* **`/verticals`**: The self-contained logic and services for specialized search verticals like Images and News.
* **`/shared_libs`**: Common libraries and utilities (e.g., networking, crypto) shared across multiple services.

***

## 🛠️ Technology Stack

This project leverages a diverse set of modern technologies to achieve its goals.

| Category          | Technologies                                                                   |
| ----------------- | ------------------------------------------------------------------------------ |
| **Backend** | **Go**, **Rust**, Python                                                       |
| **Machine Learning**| Python (PyTorch/TensorFlow), Rust (for high-performance inference)           |
| **Frontend** | JavaScript, HTML                                                               |
| **Query Parsing** | Lex/Yacc, OCaml/F#                                                             |
| **Infrastructure**| **Docker**, **Kubernetes**, Terraform, Istio                                   |
| **Databases** | Custom Distributed FS, Vector Databases, Relational DBs (conceptual)           |
| **CI/CD** | GitHub Actions (conceptual), custom build scripts, extensive testing suites    |

***

## 🚀 Getting Started

Follow these instructions to get the T3SS environment up and running on your local machine for development and testing purposes.

### Prerequisites

You must have the following tools installed on your system:
* Go (v1.21+)
* Rust (and Cargo)
* Python (v3.10+)
* Node.js (for frontend dependencies)
* Docker & Docker Compose
* Kubernetes (minikube or kind is recommended for local development)
* Terraform

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/iamqiss/T3SS.git](https://github.com/iamqiss/T3SS.git)
    cd T3SS
    ```

2.  **Configure Environment:**
    Copy the example configuration file and modify it with your local settings.
    ```bash
    cp infrastructure/config/staging_config.example.yaml infrastructure/config/local_config.yaml
    ```

3.  **Build Service Containers:**
    This command will build the Docker images for all microservices defined in the project.
    ```bash
    # This is a conceptual command; a real project would have a Makefile or build script
    make build-docker
    ```

4.  **Deploy to Local Kubernetes:**
    This will use the Kubernetes configuration files to deploy the entire stack to your local cluster.
    ```bash
    # This is a conceptual command
    make deploy-local
    ```

***

## 🧪 Running Tests

The project includes a comprehensive suite of tests to ensure code quality and system reliability.
```bash
# Run all unit, integration, and end-to-end tests
make test-all
