# Prompt Playground

## Português

Uma aplicação Streamlit para desenvolver, testar e comparar prompts em múltiplos modelos de linguagem grandes (LLMs) e endpoints, incluindo Ollama e APIs compatíveis com OpenAI.

### Funcionalidades

*   Suporte a múltiplos tipos de modelos (Ollama, OpenAI-compatível).
*   Comparação lado a lado de até 3 modelos.
*   Configuração de URL base e chave de API para cada tipo de endpoint.
*   Parâmetros de inferência globais e configuráveis (temperatura, max tokens, top_p).
*   Listagem dinâmica de modelos disponíveis (quando suportado pelo endpoint).
*   Interface bilíngue (Português, Inglês).

### Pré-requisitos

*   Python 3.10+ (testado especificamente com Python 3.12)
*   pip (gerenciador de pacotes Python)
*   Opcional: Acesso a um endpoint Ollama local ou remoto.
*   Opcional: Acesso a um endpoint compatível com OpenAI (com Chave API e URL Base, se necessário).

### Instalação

1.  Clone o repositório:
    ```bash
    git clone git@github.com:alisio/Prompt-Playground.git
    cd Prompt-Playground
    ```

2.  Crie e ative um ambiente virtual (recomendado):
    ```bash
    python -m venv venv
    # No macOS/Linux:
    source venv/bin/activate
    # No Windows:
    # venv\Scripts\activate
    ```

3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

### Uso

Execute a aplicação Streamlit (assumindo que o script principal se chama `app.py`):

```bash
streamlit run app.py
```

A aplicação estará acessível no seu navegador, geralmente em `http://localhost:8501`.

### Configuração

*   **Credenciais e Endpoints:**
    *   Na barra lateral, insira a "Ollama Base URL" (ex: `http://localhost:11434`).
    *   Insira a "Chave API OpenAI-compatível" e a "URL Base OpenAI-compatível" se estiver usando este tipo de endpoint. A chave pode ser "NA" se o endpoint não a exigir.
*   **Seleção de Modelos:**
    *   Escolha o número de modelos a comparar (1 a 3).
    *   Para cada modelo, ative-o, selecione o "Tipo de Endpoint" e escolha ou digite o nome do modelo.
*   **Parâmetros de Inferência:**
    *   Ajuste os parâmetros globais de "Temperatura", "Max Tokens" e "Top P" conforme necessário.

### Contribuição

Contribuições são bem-vindas. Por favor, abra uma *issue* para discutir mudanças ou um *pull request* com suas melhorias.

### Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

### Autor

Antonio Alisio de Meneses Cordeiro
<alisio.meneses@gmail.com>

### Testado Em

*   macOS Sequoia
*   Python 3.12

---

## English

A Streamlit application to develop, test, and compare prompts across multiple Large Language Models (LLMs) and endpoints, including Ollama and OpenAI-compatible APIs.

### Features

*   Support for multiple model types (Ollama, OpenAI-compatible).
*   Side-by-side comparison of up to 3 models.
*   Configuration of base URL and API key for each endpoint type.
*   Global and configurable inference parameters (temperature, max tokens, top_p).
*   Dynamic listing of available models (when supported by the endpoint).
*   Bilingual interface (Portuguese, English).

### Prerequisites

*   Python 3.10+ (specifically tested with Python 3.12)
*   pip (Python package manager)
*   Optional: Access to a local or remote Ollama endpoint.
*   Optional: Access to an OpenAI-compatible endpoint (with API Key and Base URL, if required).

### Installation

1.  Clone the repository:
    ```bash
    git clone git@github.com:alisio/Prompt-Playground.git
    cd Prompt-Playground
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Run the Streamlit application (assuming the main script is named `app.py`):

```bash
streamlit run app.py
```

The application will be accessible in your browser, usually at `http://localhost:8501`.

### Configuration

*   **Credentials and Endpoints:**
    *   In the sidebar, enter the "Ollama Base URL" (e.g., `http://localhost:11434`).
    *   Enter the "OpenAI-compatible API Key" and "OpenAI-compatible Base URL" if using this type of endpoint. The key can be "NA" if the endpoint does not require one.
*   **Model Selection:**
    *   Choose the number of models to compare (1 to 3).
    *   For each model, activate it, select the "Endpoint Type," and choose or enter the model name.
*   **Inference Parameters:**
    *   Adjust the global "Temperature," "Max Tokens," and "Top P" parameters as needed.

### Contributing

Contributions are welcome. Please open an issue to discuss changes or a pull request with your improvements.

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.

### Author

Antonio Alisio de Meneses Cordeiro
<alisio.meneses@gmail.com>

### Tested On

*   macOS Sequoia
*   Python 3.12