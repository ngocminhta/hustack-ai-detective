# HUSTack-AI Detective Documentation

HUSTack-AI Detective is an API built with FastAPI and Gradio designed to analyze source code and determine whether it is human-written or AI-generated. In advanced mode, it can also identify the AI model family that generated the code. The API leverages Hugging Face’s Transformers pipelines to perform text classification on the provided code after cleaning and preprocessing it.

## Features

- **AI vs. Human Detection:** Uses a text classification model to decide if the submitted code is human-written or generated by AI.
- **Advanced Model Detection:** In advanced mode, if the code is flagged as AI-generated, an additional model identifies the family of AI models (e.g., Gemini 1.x Family, GPT Family).
- **Interactive Web Interface:** Built using Gradio, enabling users to interactively test and analyze code snippets.

## Prerequisites

This project is distributed as a Docker image. You will need Docker installed on your system to run the application.

- **Docker:** Install Docker on your machine. Follow the instructions at the [Docker documentation](https://docs.docker.com/get-docker/) for your specific operating system.
- **Model Files:** The Docker image includes the required dependencies. However, ensure that the directories `./ai-detector` and `./model-detector` (mounted appropriately) contain the necessary text-classification models (compatible with Hugging Face Transformers).

## Installation

**Step 1. Clone the Repository:**

```sh
git clone https://github.com/ngocminhta/hustack-ai-detective
cd hustack-ai-detective
```

**Step 2. Build the Docker Image:**

```sh
docker build -t hustack-ai-detective .
```

or you can pull from Docker Hub:

```sh
docker pull ngocminhta/hustack-ai-detective:latest
```

**Step 3. Running the Docker Container:**

```sh
docker run -e PORT=8000 -p 8000:8000 ngocminhta/hustack-ai-detective:latest
```

   The API will be accessible at `http://localhost:8000`.

## API Endpoints

### POST `/classify`

- **Description:**  
  Classifies a list of source code snippets to determine whether they are human-written or AI-generated. In advanced mode, it can also identify the AI model family for AI-generated code.

- **Request Body (JSON):**
 
```json
{
  "code": [
    "first code snippet here",
    "second code snippet here"
  ],
  "language": [
    "Python", 
    "C++"
  ],
  "mode": "normal"  // or "advanced"
}
```

  - `code`: A list of source code strings.
  - `language`: A list of corresponding programming languages (must match the length of the `code` list). Supported values: `"C"`, `"C++"`, `"Java"`, `"Python"`.
  - `mode`: Optional. `"normal"` (default) or `"advanced"`.

- **Response:**
  - **Normal Mode:**

```json
{
  "results": [
    {"source": "Human"},
    {"source": "AI"}
  ]
}
```

  - **Advanced Mode:**

```json
{
  "results": [
    {"source": "Human"},
    {"source": "AI", "ai_model": "GPT Family"}
  ]
}
```

- **Error Responses:**
  - Missing or invalid code list:

```json
{
  "error": "No code list provided."
}
```

  - Mismatched lengths of code and language lists:
 
```json
{
  "error": "Language list must match code list length."
}
```

## Interactive Interface

The project integrates a Gradio interface for testing the classification system interactively. Also, user can read the instruction at README page ([/readme](/readme)):

- **Interface Features:**
  - **Language Selection:** Choose between C, C++, Java, and Python.
  - **Code Input:** Enter or paste the source code into the provided code box.
  - **"Check Origin" Button:** Triggers the classification process.
  - **"Clear" Button:** Resets the inputs.
  - **Results Display:** Shows the classification result and, in advanced mode, detailed model information.

When you run the application via Docker, the Gradio UI is mounted and available at the root path of the API server.

-----------
© This product is developed by [Minh N. Ta](https://tnminh.com) and contributed to the School of Information and Communication Technology, Hanoi University of Science and Technology in 2025.