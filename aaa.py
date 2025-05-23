import streamlit as st
import requests
import json

# --- Funções de Chamada aos Endpoints ---

def get_ollama_models(base_url):
    """Busca modelos disponíveis de um endpoint Ollama."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
        models_data = response.json().get("models", [])
        return sorted([model["name"] for model in models_data])
    except requests.exceptions.Timeout:
        st.error(f"Timeout ao buscar modelos Ollama de {base_url}.")
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao buscar modelos Ollama de {base_url}: {e}")
    except json.JSONDecodeError:
        st.error(f"Resposta inválida (não JSON) de {base_url}/api/tags")
    except Exception as e:
        st.error(f"Erro inesperado ao buscar modelos Ollama: {e}")
    return []

def get_openai_compatible_models(base_url, api_key=None):
    """Busca modelos disponíveis de um endpoint compatível com OpenAI."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        response = requests.get(f"{base_url.rstrip('/')}/v1/models", headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return sorted([model["id"] for model in data.get("data", []) if "id" in model])
    except requests.exceptions.Timeout:
        st.error(f"Timeout ao buscar modelos OpenAI de {base_url}.")
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao buscar modelos OpenAI de {base_url}: {e}")
    except json.JSONDecodeError:
        st.error(f"Resposta inválida (não JSON) de {base_url}/v1/models")
    except Exception as e:
        st.error(f"Erro inesperado ao buscar modelos OpenAI: {e}")
    return []

def query_ollama(base_url, model_name, prompt):
    """Envia um prompt para um modelo Ollama e retorna a resposta."""
    try:
        payload = {"model": model_name, "prompt": prompt, "stream": False}
        response = requests.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=120) # Timeout maior para geração
        response.raise_for_status()
        return response.json().get("response", "Nenhuma resposta recebida.")
    except requests.exceptions.Timeout:
        return f"Timeout ao consultar Ollama ({model_name}). Aumente o timeout se necessário."
    except requests.exceptions.RequestException as e:
        return f"Erro ao consultar Ollama ({model_name}): {e}"
    except json.JSONDecodeError:
        return f"Resposta inválida (não JSON) do Ollama ({model_name})."
    except Exception as e:
        return f"Erro inesperado ao consultar Ollama: {e}"

def query_openai_compatible(base_url, api_key, model_name, prompt):
    """Envia um prompt para um modelo compatível com OpenAI e retorna a resposta."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    try:
        response = requests.post(f"{base_url.rstrip('/')}/v1/chat/completions", headers=headers, json=payload, timeout=120) # Timeout maior
        response.raise_for_status()
        data = response.json()
        
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            first_choice = data["choices"][0]
            if "message" in first_choice and isinstance(first_choice["message"], dict):
                if "content" in first_choice["message"]:
                    return first_choice["message"]["content"]
                else:
                    return "Resposta recebida, mas 'content' não encontrado na mensagem."
            else:
                return "Resposta recebida, mas 'message' não encontrado ou formato inválido na primeira escolha ('choice')."
        else:
            return f"Resposta recebida, mas 'choices' está ausente, vazio ou formato inválido. Resposta completa: {json.dumps(data)}"

    except requests.exceptions.Timeout:
        return f"Timeout ao consultar modelo OpenAI-compatível ({model_name}). Aumente o timeout se necessário."
    except requests.exceptions.RequestException as e:
        error_message = f"Erro HTTP ao consultar modelo OpenAI-compatível ({model_name}): {e}"
        if e.response is not None:
            try:
                error_detail = e.response.json() 
                error_message += f" - Detalhe: {json.dumps(error_detail)}"
            except json.JSONDecodeError:
                error_message += f" - Detalhe (não JSON): {e.response.text[:500]}" # Limitar tamanho do texto
        return error_message
    except json.JSONDecodeError:
        return f"Resposta inválida (não JSON) do modelo OpenAI-compatível ({model_name})."
    except Exception as e:
        return f"Erro inesperado ao consultar modelo OpenAI-compatível ({model_name}): {e}"

# --- Interface Streamlit ---
st.set_page_config(layout="wide")
st.title("🚀 Comparador de Respostas de LLMs")

st.markdown("""
Esta ferramenta permite que você envie um prompt para dois modelos de linguagem diferentes
e compare suas respostas lado a lado. Configure os endpoints e modelos abaixo.
""")

if 'model_configs' not in st.session_state:
    st.session_state.model_configs = [
        {"type": "Ollama", "base_url": "http://localhost:11434", "api_key": None, "selected_model": None, "available_models": []},
        {"type": "OpenAI-compatível", "base_url": "", "api_key": None, "selected_model": None, "available_models": []}
    ]
if 'responses' not in st.session_state:
    st.session_state.responses = [None, None]
if 'prompt' not in st.session_state:
    st.session_state.prompt = "Write a short poem about Python programming."

NUM_MODELS = 2

cols_config = st.columns(NUM_MODELS)

for i in range(NUM_MODELS):
    with cols_config[i]:
        st.subheader(f"Configuração Modelo {i+1}")
        key_suffix = f"_model_{i}"
        
        # Acessar e modificar diretamente o dicionário em session_state
        current_config_key = f"model_config_{i}" # Chave única para o dicionário de configuração do modelo
        
        # Inicializar a configuração específica do modelo se não existir
        if current_config_key not in st.session_state:
            st.session_state[current_config_key] = st.session_state.model_configs[i]

        # Referenciar o dicionário no session_state
        config_ref = st.session_state.model_configs[i]


        config_ref["type"] = st.selectbox(
            f"Tipo de Endpoint",
            ["Ollama", "OpenAI-compatível"],
            index=["Ollama", "OpenAI-compatível"].index(config_ref["type"]), # Define o índice baseado no valor atual
            key=f"endpoint_type{key_suffix}"
        )

        config_ref["base_url"] = st.text_input(
            f"URL Base",
            value=config_ref["base_url"],
            placeholder="Ex: http://localhost:11434 ou https://api.example.com",
            key=f"base_url{key_suffix}"
        )

        if config_ref["type"] == "OpenAI-compatível":
            config_ref["api_key"] = st.text_input(
                f"Chave API (opcional)",
                value=config_ref.get("api_key", ""),
                type="password",
                key=f"api_key{key_suffix}"
            )
        else:
            config_ref["api_key"] = None # Resetar se não for OpenAI

        if st.button(f"Carregar Modelos do Endpoint {i+1}", key=f"load_models{key_suffix}"):
            if not config_ref["base_url"]:
                st.warning("Por favor, insira a URL Base.")
                config_ref["available_models"] = []
                config_ref["selected_model"] = None
            else:
                with st.spinner(f"Buscando modelos do Endpoint {i+1}..."):
                    if config_ref["type"] == "Ollama":
                        config_ref["available_models"] = get_ollama_models(config_ref["base_url"])
                    elif config_ref["type"] == "OpenAI-compatível":
                        config_ref["available_models"] = get_openai_compatible_models(config_ref["base_url"], config_ref["api_key"])
                    
                    if not config_ref["available_models"]:
                        st.error("Nenhum modelo encontrado ou falha ao carregar.")
                    # Resetar o modelo selecionado QUANDO novos modelos são carregados
                    config_ref["selected_model"] = None 
            st.rerun()

        options_for_selectbox = config_ref.get("available_models", [])
        if not isinstance(options_for_selectbox, list):
            options_for_selectbox = []
            
        current_selected_model_val = config_ref.get("selected_model")
        current_model_idx_val = None
        if current_selected_model_val and options_for_selectbox and current_selected_model_val in options_for_selectbox:
            current_model_idx_val = options_for_selectbox.index(current_selected_model_val)
        
        config_ref["selected_model"] = st.selectbox(
            f"Escolha o Modelo",
            options_for_selectbox,
            index=current_model_idx_val, # Usar o índice calculado
            key=f"model_select{key_suffix}", # Chave única para o selectbox
            disabled=not bool(options_for_selectbox),
            placeholder="Escolha um modelo" if options_for_selectbox else "Carregue modelos primeiro"
        )

st.session_state.prompt = st.text_area(
    "Digite seu prompt aqui:",
    value=st.session_state.prompt,
    height=150,
    key="prompt_input"
)

if st.button("Gerar Respostas", type="primary", key="generate_button"):
    if not st.session_state.prompt:
        st.warning("Por favor, insira um prompt.")
    else:
        all_configs_valid = True
        for i in range(NUM_MODELS):
            config = st.session_state.model_configs[i]
            if not (config.get("base_url") and config.get("selected_model")):
                st.error(f"Modelo {i+1} não está completamente configurado. Verifique URL Base e seleção de Modelo.")
                all_configs_valid = False
        
        if all_configs_valid:
            temp_responses = [None] * NUM_MODELS
            # Usar st.status para mensagens de log que podem ser expandidas/colapsadas
            with st.status("Gerando respostas...", expanded=True) as status_container:
                for i in range(NUM_MODELS):
                    config = st.session_state.model_configs[i]
                    status_container.write(f"Consultando Modelo {i+1}: {config['selected_model']} em {config['base_url']}...")
                    try:
                        if config["type"] == "Ollama":
                            temp_responses[i] = query_ollama(
                                config["base_url"],
                                config["selected_model"],
                                st.session_state.prompt
                            )
                        elif config["type"] == "OpenAI-compatível":
                            temp_responses[i] = query_openai_compatible(
                                config["base_url"],
                                config["api_key"],
                                config["selected_model"],
                                st.session_state.prompt
                            )
                        status_container.write(f"Resposta recebida do Modelo {i+1}.")
                    except Exception as e: 
                        error_msg = f"Falha na lógica de chamada para o Modelo {i+1} ({config['selected_model']}): {e}"
                        st.error(error_msg) # Mostrar erro principal fora do status
                        status_container.write(error_msg) # Também logar no status
                        temp_responses[i] = f"Erro interno ao processar: {e}"
                st.session_state.responses = temp_responses
                status_container.update(label="Respostas geradas!", state="complete", expanded=False)
        else:
            st.warning("Configure ambos os modelos corretamente antes de gerar respostas.")

if any(res is not None for res in st.session_state.responses):
    st.divider()
    st.subheader("Respostas dos Modelos")
    cols_responses = st.columns(NUM_MODELS)
    for i in range(NUM_MODELS):
        with cols_responses[i]:
            config = st.session_state.model_configs[i]
            model_name_display = config.get("selected_model", f"Modelo {i+1} (Não selecionado)")
            st.markdown(f"#### {model_name_display}")
            
            response_content = st.session_state.responses[i]
            if response_content:
                # Verificação de erro mais robusta
                is_error_response = False
                if isinstance(response_content, str):
                    keywords = ["erro", "failed", "não encontrado", "inválida", "ausente", "timeout", "http", "exception"]
                    is_error_response = any(keyword in response_content.lower() for keyword in keywords)

                if is_error_response:
                    st.error(response_content)
                else:
                    st.markdown(response_content, unsafe_allow_html=True)
            elif config.get("selected_model"): # Se um modelo foi selecionado mas não houve resposta
                st.info("Nenhuma resposta retornada ou ocorreu um erro não capturado.")
            # else: # Não mostrar nada se o modelo nem foi selecionado e não houve tentativa de resposta.
            #    st.empty()

st.markdown("---")
st.caption("Desenvolvido como um comparador de LLMs.")
