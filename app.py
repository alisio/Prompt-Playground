import streamlit as st
import openai
import ollama
import json # Para tratar possíveis erros de JSON em respostas
import time # Para medir o tempo de inferência

# --- Language Configuration ---
translations = {
    "page_title": {
        "pt": "Prompt-Playground: Comparador de Prompts LLM",
        "en": "Prompt-Playground: LLM Prompt Comparator"
    },
    "app_title": {
        "pt": "Prompt-Playground: Painel Comparador de Prompts LLM",
        "en": "Prompt-Playground: LLM Prompt Comparator Dashboard"
    },
    "app_subtitle": {
        "pt": "Desenvolva e compare prompts em diferentes modelos e endpoints LLM.",
        "en": "Develop and compare prompts across different LLM models and endpoints."
    },
    "general_settings": {
        "pt": "Configurações Gerais",
        "en": "General Settings"
    },
    "credentials_endpoints": {
        "pt": "Credenciais e Endpoints",
        "en": "Credentials and Endpoints"
    },
    "ollama_base_url": {
        "pt": "Ollama Base URL",
        "en": "Ollama Base URL"
    },
    "openai_api_key": {
        "pt": "Chave API OpenAI-compatível",
        "en": "OpenAI-compatible API Key"
    },
    "openai_api_key_help": {
        "pt": "Pode ser 'NA' ou qualquer string se o endpoint não exigir chave.",
        "en": "Can be 'NA' or any string if the endpoint doesn't require a key."
    },
    "openai_base_url": {
        "pt": "URL Base OpenAI-compatível",
        "en": "OpenAI-compatible Base URL"
    },
    "openai_base_url_help": {
        "pt": "Ex: https://api.openai.com/v1 ou URL do seu endpoint privado",
        "en": "Ex: https://api.openai.com/v1 or your private endpoint URL"
    },
    "inference_parameters": {
        "pt": "Parâmetros de Inferência",
        "en": "Inference Parameters"
    },
    "temperature": {
        "pt": "Temperatura",
        "en": "Temperature"
    },
    "max_tokens": {
        "pt": "Max Tokens",
        "en": "Max Tokens"
    },
    "top_p": {
        "pt": "Top P",
        "en": "Top P"
    },
    "compare_models": {
        "pt": "Comparar Modelos:",
        "en": "Compare Models:"
    },
    "model_config_header": {
        "pt": "Modelo {number}",
        "en": "Model {number}"
    },
    "activate_model": {
        "pt": "Ativar Modelo {number}",
        "en": "Activate Model {number}"
    },
    "endpoint_type": {
        "pt": "Tipo de Endpoint (Modelo {number})",
        "en": "Endpoint Type (Model {number})"
    },
    "ollama_model_select": {
        "pt": "Modelo Ollama (Modelo {number})",
        "en": "Ollama Model (Model {number})"
    },
    "ollama_model_text": {
        "pt": "Nome do Modelo Ollama (Modelo {number})",
        "en": "Ollama Model Name (Model {number})"
    },
    "ollama_model_text_help": {
        "pt": "Ex: llama3:latest. Modelos não puderam ser listados.",
        "en": "Ex: llama3:latest. Models could not be listed."
    },
    "openai_model_select": {
        "pt": "Modelo OpenAI (Modelo {number})",
        "en": "OpenAI Model (Model {number})"
    },
    "openai_model_select_help": {
        "pt": "Selecione ou digite o nome do modelo se não estiver na lista.",
        "en": "Select or type the model name if not in the list."
    },
    "openai_type_manual_toggle": {
        "pt": "Digitar nome do modelo OpenAI manualmente (Modelo {number})",
        "en": "Manually type OpenAI model name (Model {number})"
    },
    "openai_model_text": {
        "pt": "Nome do Modelo OpenAI (Modelo {number})",
        "en": "OpenAI Model Name (Model {number})"
    },
    "openai_model_text_help": {
        "pt": "Ex: gpt-4, gpt-3.5-turbo, ou nome do seu modelo privado",
        "en": "Ex: gpt-4, gpt-3.5-turbo, or your private model name"
    },
    "openai_model_text_alt_help": {
        "pt": "Ex: gpt-4, gpt-3.5-turbo, ou nome do seu modelo privado. Modelos não puderam ser listados.",
        "en": "Ex: gpt-4, gpt-3.5-turbo, or your private model name. Models could not be listed."
    },
    "prompt_area_header": {
        "pt": "Prompt",
        "en": "Prompt"
    },
    "prompt_area_label": {
        "pt": "Digite seu prompt aqui:",
        "en": "Enter your prompt here:"
    },
    "default_prompt_value": {
        "pt": "Escreva um poema sobre Python",
        "en": "Write a poem about Python"
    },
    "generate_button": {
        "pt": "Gerar Respostas",
        "en": "Generate Responses"
    },
    "warning_empty_prompt": {
        "pt": "Por favor, insira um prompt.",
        "en": "Please enter a prompt."
    },
    "spinner_generating": {
        "pt": "Gerando respostas...",
        "en": "Generating responses..."
    },
    "error_model_not_specified": {
        "pt": "Modelo {number}: Nome do modelo não especificado.",
        "en": "Model {number}: Model name not specified."
    },
    "error_ollama_url_not_set": {
        "pt": "Modelo {number} (Ollama): URL base do Ollama não configurada.",
        "en": "Model {number} (Ollama): Ollama base URL not configured."
    },
    "error_openai_creds_not_set": {
        "pt": "Modelo {number} (OpenAI): API Key ou Base URL não configurados.",
        "en": "Model {number} (OpenAI): API Key or Base URL not configured."
    },
    "success_responses_generated": {
        "pt": "Respostas geradas!",
        "en": "Responses generated!"
    },
    "comparison_results_header": {
        "pt": "Resultados da Comparação",
        "en": "Comparison Results"
    },
    "model_display_name_header": {
        "pt": "Modelo {number}: {name}",
        "en": "Model {number}: {name}"
    },
    "model_not_configured": {
        "pt": "Modelo {number} (Não Configurado)",
        "en": "Model {number} (Not Configured)"
    },
    "model_endpoint_type_display": {
        "pt": "Modelo {number} ({endpoint_type})",
        "en": "Model {number} ({endpoint_type})"
    },
    "endpoint_label": {
        "pt": "Endpoint:",
        "en": "Endpoint:"
    },
    "not_applicable_abbrev": {
        "pt": "N/A",
        "en": "N/A"
    },
    "parameters_used_expander": {
        "pt": "Parâmetros Usados",
        "en": "Parameters Used"
    },
    "parameters_caption": {
        "pt": "Temp: {temp}, Max Tokens: {max_tokens}, Top P: {top_p}",
        "en": "Temp: {temp}, Max Tokens: {max_tokens}, Top P: {top_p}"
    },
    "response_from_model_label": {
        "pt": "Resposta de {model_name}",
        "en": "Response from {model_name}"
    },
    "info_no_active_models": {
        "pt": "Nenhum modelo ativo para exibir resultados. Configure e ative modelos na barra lateral.",
        "en": "No active models to display results. Configure and activate models in the sidebar."
    },
    "info_click_generate": {
        "pt": "Clique em 'Gerar Respostas' após configurar os modelos e inserir um prompt.",
        "en": "Click 'Generate Responses' after configuring models and entering a prompt."
    },
    "how_to_use_header": {
        "pt": "Como usar:",
        "en": "How to use:"
    },
    "how_to_use_content": {
        "pt": """
1.  Configure as URLs e chaves de API na seção "Credenciais".
2.  Escolha quantos modelos comparar (1 a 3).
3.  Para cada modelo:
    * Ative-o.
    * Selecione o tipo de endpoint.
    * Escolha/digite o nome do modelo.
    * (Opcional) Ajuste parâmetros de inferência (padrão globais).
4.  Digite seu prompt na área principal.
5.  Clique em "Gerar Respostas".
""",
        "en": """
1.  Configure API URLs and keys in the "Credentials" section.
2.  Choose how many models to compare (1 to 3).
3.  For each model:
    * Activate it.
    * Select the endpoint type.
    * Choose/enter the model name.
    * (Optional) Adjust inference parameters (global defaults).
4.  Enter your prompt in the main area.
5.  Click "Generate Responses".
"""
    },
    "error_listing_ollama_models": {
        "pt": "Ollama ({base_url}): Erro ao listar modelos: {e}",
        "en": "Ollama ({base_url}): Error listing models: {e}"
    },
    "error_querying_ollama": {
        "pt": "Erro ao consultar Ollama ({model_name}): {e}",
        "en": "Error querying Ollama ({model_name}): {e}"
    },
    "fetching_models_from": {
        "pt": "Buscando modelos de {base_url}...",
        "en": "Fetching models from {base_url}..."
    },
    "error_openai_connection": {
        "pt": "Erro de conexão com API OpenAI ({model_name}): {e}",
        "en": "OpenAI API connection error ({model_name}): {e}"
    },
    "error_openai_auth": {
        "pt": "Erro de autenticação com API OpenAI ({model_name}): Chave inválida ou não fornecida? {e}",
        "en": "OpenAI API authentication error ({model_name}): Invalid or missing key? {e}"
    },
    "error_openai_ratelimit": {
        "pt": "Erro de limite de taxa com API OpenAI ({model_name}): {e}",
        "en": "OpenAI API rate limit error ({model_name}): {e}"
    },
    "error_openai_notfound": {
        "pt": "Erro: Modelo '{model_name}' não encontrado no endpoint '{base_url}'. Verifique o nome. {e}",
        "en": "Error: Model '{model_name}' not found at endpoint '{base_url}'. Check the name. {e}"
    },
    "error_querying_openai_api": {
        "pt": "Erro ao consultar API OpenAI ({model_name}): {msg}",
        "en": "Error querying OpenAI API ({model_name}): {msg}"
    },
    "language": {
        "pt": "Idioma",
        "en": "Language"
    },
    "inference_time_label": {
        "pt": "Tempo de Inferência",
        "en": "Inference Time"
    },
    "input_tokens_label": {
        "pt": "Tokens de Entrada",
        "en": "Input Tokens"
    },
    "output_tokens_label": {
        "pt": "Tokens de Saída",
        "en": "Output Tokens"
    },
    "tokens_unit": {
        "pt": "tokens",
        "en": "tokens"
    },
    "seconds_unit_short": {
        "pt": "s",
        "en": "s"
    },
    "inference_metrics_header": {
        "pt": "Métricas de Inferência",
        "en": "Inference Metrics"
    },
}


if 'lang' not in st.session_state:
    st.session_state.lang = "pt" # Default language

def t(key):
    return translations.get(key, {}).get(st.session_state.lang, f"[{key}_{st.session_state.lang}]")

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title=t("page_title"))

# --- Sidebar Language Selector ---
with st.sidebar:
    selected_lang_display = "Português" if st.session_state.lang == "pt" else "English"
    lang_options = {"Português": "pt", "English": "en"}
    
    def format_func(option): # To display "Português" or "English" in selectbox
        return option

    new_lang_display = st.selectbox(
        t("language"),
        options=list(lang_options.keys()),
        index=list(lang_options.values()).index(st.session_state.lang),
        format_func=format_func, # Shows "Português" or "English"
        key="lang_selector_display"
    )
    new_lang_code = lang_options[new_lang_display]

    if st.session_state.lang != new_lang_code:
        st.session_state.lang = new_lang_code
        st.rerun()


# --- Helper Functions for API Calls ---

def get_ollama_models(base_url):
    """Lists available models from an Ollama endpoint."""
    try:
        client = ollama.Client(host=base_url)
        models_info = client.list()
        return [model['model'] for model in models_info['models']]
    except Exception as e:
        st.sidebar.error(t("error_listing_ollama_models").format(base_url=base_url, e=e))
        return []

def query_ollama(base_url, model_name, prompt, temperature, max_tokens, top_p):
    """Sends a prompt to an Ollama model and returns detailed response info."""
    start_time = time.time()
    try:
        client = ollama.Client(host=base_url)
        response = client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': temperature,
                'num_predict': max_tokens, # num_predict is often used for max_tokens in Ollama
                'top_p': top_p,
            }
        )
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Ollama response structure for non-streaming:
        # response = {
        #   'model': 'llama2:7b', 'created_at': '...', 'message': {'role': 'assistant', 'content': '...'},
        #   'done': True, 'total_duration': ..., 'load_duration': ..., 'prompt_eval_count': X,
        #   'prompt_eval_duration': ..., 'eval_count': Y, 'eval_duration': ...
        # }
        return {
            "text": response['message']['content'],
            "time": inference_time,
            "in_tokens": response.get('prompt_eval_count'),
            "out_tokens": response.get('eval_count'),
            "raw_response": response
        }
    except Exception as e:
        end_time = time.time()
        return {
            "text": t("error_querying_ollama").format(model_name=model_name, e=e),
            "time": end_time - start_time,
            "in_tokens": None,
            "out_tokens": None,
            "raw_response": {"error": str(e)}
        }

def get_openai_compatible_models(api_key, base_url):
    """
    Tries to list models from an OpenAI-compatible endpoint.
    Many private endpoints may not support this or require specific permissions.
    """
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception:
        return []


def query_openai_compatible(api_key, base_url, model_name, prompt, temperature, max_tokens, top_p):
    """Sends a prompt to an OpenAI-compatible endpoint and returns detailed response info."""
    start_time = time.time()
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        end_time = time.time()
        inference_time = end_time - start_time
        
        in_tokens = None
        out_tokens = None
        if completion.usage:
            in_tokens = completion.usage.prompt_tokens
            out_tokens = completion.usage.completion_tokens
            
        return {
            "text": completion.choices[0].message.content,
            "time": inference_time,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "raw_response": completion.model_dump_json(indent=2) # Pydantic v2
        }
    except openai.APIConnectionError as e:
        msg = t("error_openai_connection").format(model_name=model_name, e=e)
    except openai.AuthenticationError as e:
        msg = t("error_openai_auth").format(model_name=model_name, e=e)
    except openai.RateLimitError as e:
        msg = t("error_openai_ratelimit").format(model_name=model_name, e=e)
    except openai.NotFoundError as e:
        msg = t("error_openai_notfound").format(model_name=model_name, base_url=base_url, e=e)
    except Exception as e:
        try:
            # Attempt to parse a JSON error response, common with OpenAI-like APIs
            error_body = getattr(e, 'response', None) # Get response if available
            if error_body:
                error_data = error_body.json()
                err_msg_detail = error_data.get("error", {}).get("message", str(e))
            else: # Fallback if no 'response' attribute or not JSON
                 err_msg_detail = str(e)
            msg = t("error_querying_openai_api").format(model_name=model_name, msg=err_msg_detail)
        except json.JSONDecodeError: # If e.response.json() fails
            msg = t("error_querying_openai_api").format(model_name=model_name, msg=str(e))
        except AttributeError: # If e.response does not exist
             msg = t("error_querying_openai_api").format(model_name=model_name, msg=str(e))


    end_time = time.time() # Record time even for errors
    return {
        "text": msg,
        "time": end_time - start_time,
        "in_tokens": None,
        "out_tokens": None,
        "raw_response": {"error": msg, "details": str(e)}
    }


# --- Streamlit Interface ---
st.title(t("app_title"))
st.markdown(t("app_subtitle"))

# --- Session State for Storing Responses and Details ---
if 'response_details' not in st.session_state:
    st.session_state.response_details = [None, None, None] # Will store dicts with text, time, tokens
if 'model_configs' not in st.session_state:
    st.session_state.model_configs = [{}, {}, {}]

# --- Sidebar for Global and Model-specific Settings ---
with st.sidebar:
    st.header(t("general_settings"))

    st.subheader(t("credentials_endpoints"))
    ollama_base_url = st.text_input(t("ollama_base_url"), value="http://localhost:11434", key="ollama_url")
    
    openai_api_key = st.text_input(
        t("openai_api_key"), 
        type="password", 
        key="openai_key", 
        help=t("openai_api_key_help")
    )
    openai_base_url = st.text_input(
        t("openai_base_url"), 
        key="openai_base_url",
        value="https://api.openai.com/v1",
        help=t("openai_base_url_help")
    )

    st.subheader(t("inference_parameters"))
    global_temperature = st.slider(t("temperature"), 0.0, 2.0, 0.7, 0.05, key="global_temp") # OpenAI allows up to 2.0
    global_max_tokens = st.number_input(t("max_tokens"), 50, 16384, 512, 50, key="global_max_tokens") # Increased upper limit
    global_top_p = st.slider(t("top_p"), 0.0, 1.0, 0.9, 0.05, key="global_top_p")

    st.markdown("---")
    
    num_models_to_compare = st.radio(
        t("compare_models"),
        (1, 2, 3),
        index=0, # Default to 1 model for simplicity on first load
        horizontal=True,
        key="num_models"
    )
    st.markdown("---")

    model_configs_ui = []
    available_ollama_models = get_ollama_models(ollama_base_url) if ollama_base_url else []
    
    # Initialize session state for OpenAI models cache and last URL
    if 'last_openai_base_url' not in st.session_state:
        st.session_state.last_openai_base_url = ""
    if 'last_openai_api_key' not in st.session_state: # Also consider API key changes
        st.session_state.last_openai_api_key = ""
    if 'available_openai_models' not in st.session_state:
        st.session_state.available_openai_models = []

    # Fetch OpenAI models if URL or Key changed and are valid
    if openai_base_url and openai_api_key and \
       (openai_base_url != st.session_state.get('last_openai_base_url', "") or \
        openai_api_key != st.session_state.get('last_openai_api_key', "")):
        with st.spinner(t("fetching_models_from").format(base_url=openai_base_url)):
            st.session_state.available_openai_models = get_openai_compatible_models(openai_api_key, openai_base_url)
            st.session_state.last_openai_base_url = openai_base_url
            st.session_state.last_openai_api_key = openai_api_key # Store current key
    
    available_openai_models_cached = st.session_state.available_openai_models

    for i in range(num_models_to_compare):
        st.header(t("model_config_header").format(number=i+1))
        config = {}
        config['active'] = st.checkbox(t("activate_model").format(number=i+1), value=True if i==0 else False, key=f"active_{i}") # First model active by default
        
        if config['active']:
            config['endpoint_type'] = st.selectbox(
                t("endpoint_type").format(number=i+1),
                ("Ollama", "OpenAI-compatible"), 
                key=f"endpoint_type_{i}"
            )

            if config['endpoint_type'] == "Ollama":
                if available_ollama_models:
                    default_ollama_model = available_ollama_models[0] if available_ollama_models else None
                    config['model_name'] = st.selectbox(
                        t("ollama_model_select").format(number=i+1),
                        options=available_ollama_models,
                        index=available_ollama_models.index(default_ollama_model) if default_ollama_model else 0,
                        key=f"ollama_model_{i}"
                    )
                else:
                    config['model_name'] = st.text_input(
                        t("ollama_model_text").format(number=i+1),
                        help=t("ollama_model_text_help"),
                        placeholder="e.g., llama3:latest",
                        key=f"ollama_model_text_{i}"
                    )
            else: # OpenAI-compatível
                # Manually type toggle
                manual_type_openai_key = f"openai_manual_toggle_{i}"
                if manual_type_openai_key not in st.session_state:
                    st.session_state[manual_type_openai_key] = not bool(available_openai_models_cached)


                st.session_state[manual_type_openai_key] = st.checkbox(
                    t("openai_type_manual_toggle").format(number=i+1),
                    value=st.session_state[manual_type_openai_key],
                    key=manual_type_openai_key + "_widget" # Ensure unique widget key
                )
                
                if not st.session_state[manual_type_openai_key] and available_openai_models_cached:
                    default_openai_model = available_openai_models_cached[0] if available_openai_models_cached else None
                    config['model_name'] = st.selectbox(
                        t("openai_model_select").format(number=i+1),
                        options=available_openai_models_cached,
                        index=available_openai_models_cached.index(default_openai_model) if default_openai_model else 0,
                        help=t("openai_model_select_help"),
                        key=f"openai_model_select_dd_{i}" # Unique key for selectbox
                    )
                else:
                     config['model_name'] = st.text_input(
                        t("openai_model_text").format(number=i+1),
                        value=config.get('model_name', "gpt-3.5-turbo"),
                        help=t("openai_model_text_alt_help") if not available_openai_models_cached else t("openai_model_text_help"),
                        placeholder="e.g., gpt-4o",
                        key=f"openai_model_text_input_{i}" # Unique key for text input
                    )

            config['temperature'] = global_temperature
            config['max_tokens'] = global_max_tokens
            config['top_p'] = global_top_p
        
        model_configs_ui.append(config)
        st.markdown("---")
    
    st.session_state.model_configs = model_configs_ui


# --- Main Area for Prompt and Responses ---
st.subheader(t("prompt_area_header"))
prompt_text = st.text_area(
    t("prompt_area_label"), 
    height=150, 
    key="prompt_input", 
    value=t("default_prompt_value")
)

if st.button(t("generate_button"), type="primary"):
    if not prompt_text.strip():
        st.warning(t("warning_empty_prompt"))
    else:
        st.session_state.response_details = [None] * 3 # Reset details
        active_configs_with_indices = [
            (idx, conf) for idx, conf in enumerate(st.session_state.model_configs[:num_models_to_compare]) 
            if conf.get('active', False)
        ]
        
        if not active_configs_with_indices:
            st.warning(t("info_no_active_models")) # Or a more specific warning
        else:
            with st.spinner(t("spinner_generating")):
                for original_idx, config in active_configs_with_indices:
                    # Validate model name presence
                    if not config.get('model_name', '').strip():
                        st.session_state.response_details[original_idx] = {
                            "text": t("error_model_not_specified").format(number=original_idx+1),
                            "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}
                        }
                        continue

                    response_data = {}
                    if config['endpoint_type'] == "Ollama":
                        if not ollama_base_url:
                            response_data = {
                                "text": t("error_ollama_url_not_set").format(number=original_idx+1),
                                "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}
                            }
                        else:
                            response_data = query_ollama(
                                ollama_base_url,
                                config['model_name'],
                                prompt_text,
                                config['temperature'],
                                config['max_tokens'],
                                config['top_p']
                            )
                    elif config['endpoint_type'] == "OpenAI-compatible":
                        if not openai_api_key or not openai_base_url:
                             response_data = {
                                "text": t("error_openai_creds_not_set").format(number=original_idx+1),
                                "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}
                            }
                        else:
                            response_data = query_openai_compatible(
                                openai_api_key,
                                openai_base_url,
                                config['model_name'],
                                prompt_text,
                                config['temperature'],
                                config['max_tokens'],
                                config['top_p']
                            )
                    st.session_state.response_details[original_idx] = response_data
            st.success(t("success_responses_generated"))

# --- Display Responses ---
st.subheader(t("comparison_results_header"))

# Filter for active models up to num_models_to_compare that have details
active_model_indices_with_details = [
    i for i, details in enumerate(st.session_state.response_details[:num_models_to_compare])
    if details is not None and st.session_state.model_configs[i].get('active', False)
]


if active_model_indices_with_details:
    cols = st.columns(len(active_model_indices_with_details))
    
    col_idx = 0
    for i in active_model_indices_with_details:
        config = st.session_state.model_configs[i]
        response_detail = st.session_state.response_details[i]
        
        # response_detail should always be a dict here due to previous logic
        response_text = response_detail.get('text', t("not_applicable_abbrev"))
        inference_time = response_detail.get('time')
        in_tokens = response_detail.get('in_tokens')
        out_tokens = response_detail.get('out_tokens')

        with cols[col_idx]:
            model_name_from_config = config.get('model_name', '')
            
            if not model_name_from_config: 
                model_display_name = t("model_not_configured").format(number=i+1)
            else:
                model_display_name = model_name_from_config
            
            st.markdown(f"#### {t('model_display_name_header').format(number=i+1, name=model_display_name)}")
            st.markdown(f"**{t('endpoint_label')}** `{config.get('endpoint_type', t('not_applicable_abbrev'))}`")
            
            # Display Inference Metrics
            st.markdown(f"**{t('inference_metrics_header')}**")
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric(
                    label=t("inference_time_label"),
                    value=f"{inference_time:.2f} {t('seconds_unit_short')}" if inference_time is not None else t("not_applicable_abbrev")
                )
            with metric_cols[1]:
                st.metric(
                    label=t("input_tokens_label"),
                    # value=f"{in_tokens} {t('tokens_unit')}" if in_tokens is not None else t("not_applicable_abbrev")
                    value=f"{in_tokens}" if in_tokens is not None else t("not_applicable_abbrev")
                )
            with metric_cols[2]:
                st.metric(
                    label=t("output_tokens_label"),
                    # value=f"{out_tokens} {t('tokens_unit')}" if out_tokens is not None else t("not_applicable_abbrev")
                    value=f"{out_tokens}" if out_tokens is not None else t("not_applicable_abbrev")
                )

            with st.expander(t("parameters_used_expander")):
                st.caption(
                    t("parameters_caption").format(
                        temp=config.get('temperature', global_temperature),
                        max_tokens=config.get('max_tokens', global_max_tokens),
                        top_p=config.get('top_p', global_top_p)
                    )
                )
                # Optionally display raw response for debugging (can be long)
                # if response_detail.get("raw_response"):
                # st.json(response_detail["raw_response"], expanded=False)


            st.text_area(
                label=t("response_from_model_label").format(model_name=model_display_name),
                value=str(response_text), # Ensure it's a string
                height=350, # Adjusted height
                key=f"response_output_{i}",
                disabled=True,
                label_visibility="collapsed"
            )
        col_idx += 1
    
elif any(st.session_state.model_configs[i].get('active') for i in range(num_models_to_compare)):
     st.info(t("info_click_generate"))
else:
    st.info(t("info_no_active_models"))


# --- Usage Tips ---
st.sidebar.markdown("---")
st.sidebar.info(f"""
**{t("how_to_use_header")}**
{t("how_to_use_content")}
""")