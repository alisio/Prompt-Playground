import streamlit as st
import openai
import ollama
import json # Para tratar possíveis erros de JSON em respostas
import time # Para medir o tempo de inferência
import requests # Para chamadas OAuth

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
    "openai_auth_method_label": {
        "pt": "Método de Autenticação OpenAI-compatível",
        "en": "OpenAI-compatible Authentication Method"
    },
    "api_key_option": {
        "pt": "Chave de API",
        "en": "API Key"
    },
    "oauth_option": {
        "pt": "OAuth (Client Credentials)",
        "en": "OAuth (Client Credentials)"
    },
    "openai_api_key": {
        "pt": "Chave API OpenAI-compatível",
        "en": "OpenAI-compatible API Key"
    },
    "openai_api_key_help": {
        "pt": "Pode ser 'NA' ou qualquer string se o endpoint não exigir chave e o método for Chave API.",
        "en": "Can be 'NA' or any string if the endpoint doesn't require a key and method is API Key."
    },
    "openai_client_id_label": {
        "pt": "Client ID (OAuth)",
        "en": "Client ID (OAuth)"
    },
    "openai_client_secret_label": {
        "pt": "Client Secret (OAuth)",
        "en": "Client Secret (OAuth)"
    },
    "openai_token_url_label": {
        "pt": "URL do Token (OAuth)",
        "en": "Token URL (OAuth)"
    },
    "openai_token_url_help": {
        "pt": "Ex: https://seu-servidor-auth.com/oauth/token",
        "en": "Ex: https://your-auth-server.com/oauth/token"
    },
    "get_oauth_token_button": {
        "pt": "Obter Token OAuth",
        "en": "Get OAuth Token"
    },
    "oauth_token_status_label": {
        "pt": "Status do Token OAuth:",
        "en": "OAuth Token Status:"
    },
    "oauth_token_success": {
        "pt": "Token OAuth obtido com sucesso!",
        "en": "OAuth Token obtained successfully!"
    },
    "oauth_token_missing_creds": {
        "pt": "Preencha Client ID, Client Secret e URL do Token.",
        "en": "Please fill in Client ID, Client Secret, and Token URL."
    },
    "oauth_token_error_fetching": {
        "pt": "Erro ao obter token OAuth: {error}",
        "en": "Error fetching OAuth token: {error}"
    },
    "error_oauth_token_unavailable": {
        "pt": "Token OAuth não disponível ou inválido. Obtenha um novo token na barra lateral.",
        "en": "OAuth token not available or invalid. Please fetch a new token in the sidebar."
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
        "pt": "Modelo OpenAI-compatível (Modelo {number})",
        "en": "OpenAI-compatible Model (Model {number})"
    },
    "openai_model_select_help": {
        "pt": "Selecione ou digite o nome do modelo se não estiver na lista.",
        "en": "Select or type the model name if not in the list."
    },
    "openai_type_manual_toggle": {
        "pt": "Digitar nome do modelo OpenAI-compatível manualmente (Modelo {number})",
        "en": "Manually type OpenAI-compatible model name (Model {number})"
    },
    "openai_model_text": {
        "pt": "Nome do Modelo OpenAI-compatível (Modelo {number})",
        "en": "OpenAI-compatible Model Name (Model {number})"
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
        "pt": "Escreva um poema sobre Python e IA",
        "en": "Write a poem about Python and AI"
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
    "error_openai_base_url_not_set": {
        "pt": "Modelo {number} (OpenAI-compatível): URL Base não configurada.",
        "en": "Model {number} (OpenAI-compatible): Base URL not configured."
    },
    "error_openai_apikey_not_set": {
        "pt": "Modelo {number} (OpenAI-compatível/API Key): Chave API não configurada.",
        "en": "Model {number} (OpenAI-compatible/API Key): API Key not configured."
    },
    "error_openai_oauth_token_not_set": {
        "pt": "Modelo {number} (OpenAI-compatível/OAuth): Token OAuth não configurado. Obtenha um na barra lateral.",
        "en": "Model {number} (OpenAI-compatible/OAuth): OAuth Token not configured. Obtain one in the sidebar."
    },
    "error_creating_openai_client_api_key": {
        "pt": "Modelo {number} (OpenAI-compatível/API Key): Erro ao criar cliente OpenAI: {e}",
        "en": "Model {number} (OpenAI-compatible/API Key): Error creating OpenAI client: {e}"
    },
    "error_creating_openai_client_oauth": {
        "pt": "Modelo {number} (OpenAI-compatível/OAuth): Erro ao criar cliente OpenAI: {e}",
        "en": "Model {number} (OpenAI-compatible/OAuth): Error creating OpenAI client: {e}"
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
1.  Configure as URLs e credenciais na seção "Credenciais e Endpoints".
    *   Para OpenAI-compatível: Escolha o método (Chave API ou OAuth).
    *   Se OAuth: Preencha Client ID, Client Secret, URL do Token e clique em "Obter Token OAuth".
2.  Escolha quantos modelos comparar (1 a 3).
3.  Para cada modelo:
    * Ative-o.
    * Selecione o tipo de endpoint.
    * Escolha/digite o nome do modelo.
4.  Digite seu prompt na área principal.
5.  Clique em "Gerar Respostas".
""",
        "en": """
1.  Configure URLs and credentials in the "Credentials and Endpoints" section.
    *   For OpenAI-compatible: Choose method (API Key or OAuth).
    *   If OAuth: Fill in Client ID, Client Secret, Token URL, and click "Get OAuth Token".
2.  Choose how many models to compare (1 to 3).
3.  For each model:
    * Activate it.
    * Select the endpoint type.
    * Choose/enter the model name.
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
        "pt": "Erro de conexão com API OpenAI-compatível ({model_name}): {e}",
        "en": "OpenAI-compatible API connection error ({model_name}): {e}"
    },
    "error_openai_auth_apikey": {
        "pt": "Erro de autenticação com API OpenAI-compatível ({model_name}): Chave inválida ou não fornecida? {e}",
        "en": "OpenAI-compatible API authentication error ({model_name}): Invalid or missing key? {e}"
    },
    "error_openai_auth_oauth": {
        "pt": "Erro de autenticação com API OpenAI-compatível ({model_name}) via OAuth: Token inválido, expirado ou permissões insuficientes? {e}",
        "en": "OpenAI-compatible API authentication error ({model_name}) via OAuth: Invalid, expired token, or insufficient permissions? {e}"
    },
    "error_openai_ratelimit": {
        "pt": "Erro de limite de taxa com API OpenAI-compatível ({model_name}): {e}",
        "en": "OpenAI-compatible API rate limit error ({model_name}): {e}"
    },
    "error_openai_notfound": {
        "pt": "Erro: Modelo '{model_name}' não encontrado no endpoint '{base_url}'. Verifique o nome. {e}",
        "en": "Error: Model '{model_name}' not found at endpoint '{base_url}'. Check the name. {e}"
    },
    "error_querying_openai_api": {
        "pt": "Erro ao consultar API OpenAI-compatível ({model_name}): {msg}",
        "en": "Error querying OpenAI-compatible API ({model_name}): {msg}"
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
    "sidebar_openai_section_header": {
        "pt": "Endpoint OpenAI-compatível",
        "en": "OpenAI-compatible Endpoint"
    }
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
    
    def format_func_lang(option):
        return option

    new_lang_display = st.selectbox(
        t("language"),
        options=list(lang_options.keys()),
        index=list(lang_options.values()).index(st.session_state.lang),
        format_func=format_func_lang,
        key="lang_selector_display_widget"
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
        st.sidebar.error(t("error_listing_ollama_models").format(base_url=base_url, e=str(e)))
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
                'num_predict': max_tokens,
                'top_p': top_p,
            }
        )
        end_time = time.time()
        inference_time = end_time - start_time
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
            "text": t("error_querying_ollama").format(model_name=model_name, e=str(e)),
            "time": end_time - start_time,
            "in_tokens": None,
            "out_tokens": None,
            "raw_response": {"error": str(e)}
        }

def get_oauth_token(token_url, client_id, client_secret):
    """Fetches an OAuth token using client credentials grant."""
    try:
        payload = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret
        }
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        response = requests.post(token_url, data=payload, headers=headers, timeout=15)
        response.raise_for_status()
        token_data = response.json()
        if "access_token" not in token_data:
            return None, "OAuth token response did not contain 'access_token'. Received: " + str(token_data)
        return token_data, None
    except requests.exceptions.HTTPError as e:
        error_detail = str(e)
        if e.response is not None:
            try:
                error_detail += f" - Response: {e.response.json()}"
            except json.JSONDecodeError:
                error_detail += f" - Response: {e.response.text}"
        return None, f"HTTPError: {error_detail}"
    except requests.exceptions.RequestException as e:
        return None, f"RequestException: {str(e)}"
    except json.JSONDecodeError as e: # Catches error if response.json() fails
        return None, f"JSONDecodeError: Failed to parse token response. {str(e)}. Response text: {response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}"
    except Exception as e:
        return None, f"Generic error: {str(e)}"

def get_openai_compatible_models(client: openai.OpenAI):
    """Lists models from an OpenAI-compatible endpoint using a pre-configured client."""
    try:
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception:
        # Errors will be handled by the calling function's spinner/error message
        return []

def query_openai_compatible(client: openai.OpenAI, model_name: str, prompt: str, temperature: float, max_tokens: int, top_p: float):
    """Sends a prompt to an OpenAI-compatible endpoint using a pre-configured client."""
    start_time = time.time()
    auth_method = st.session_state.get("openai_auth_method", "api_key") # For error message context

    try:
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
            "raw_response": completion.model_dump_json(indent=2)
        }
    except openai.APIConnectionError as e:
        msg = t("error_openai_connection").format(model_name=model_name, e=str(e))
    except openai.AuthenticationError as e:
        if auth_method == "oauth":
            msg = t("error_openai_auth_oauth").format(model_name=model_name, e=str(e))
        else:
            msg = t("error_openai_auth_apikey").format(model_name=model_name, e=str(e))
    except openai.RateLimitError as e:
        msg = t("error_openai_ratelimit").format(model_name=model_name, e=str(e))
    except openai.NotFoundError as e:
        base_url_str = str(client.base_url) if client else t("not_applicable_abbrev")
        msg = t("error_openai_notfound").format(model_name=model_name, base_url=base_url_str, e=str(e))
    except Exception as e:
        try:
            error_body = getattr(e, 'response', None)
            if error_body:
                error_data = error_body.json()
                err_msg_detail = error_data.get("error", {}).get("message", str(e))
            else:
                err_msg_detail = str(e)
            msg = t("error_querying_openai_api").format(model_name=model_name, msg=err_msg_detail)
        except json.JSONDecodeError:
            msg = t("error_querying_openai_api").format(model_name=model_name, msg=str(e))
        except AttributeError:
             msg = t("error_querying_openai_api").format(model_name=model_name, msg=str(e))

    end_time = time.time()
    return {
        "text": msg,
        "time": end_time - start_time,
        "in_tokens": None,
        "out_tokens": None,
        "raw_response": {"error": msg, "details": str(e) if 'e' in locals() else "Unknown error before exception assignment"}
    }

# --- Streamlit Interface ---
st.title(t("app_title"))
st.markdown(t("app_subtitle"))

# --- Session State Initialization ---
if 'response_details' not in st.session_state:
    st.session_state.response_details = [None, None, None]
if 'model_configs' not in st.session_state:
    st.session_state.model_configs = [{}, {}, {}]

# OpenAI Auth related session state
if 'openai_auth_method' not in st.session_state:
    st.session_state.openai_auth_method = "api_key"
if 'openai_api_key' not in st.session_state: # For API key method
    st.session_state.openai_api_key = ""
if 'openai_client_id' not in st.session_state: # For OAuth
    st.session_state.openai_client_id = ""
if 'openai_client_secret' not in st.session_state: # For OAuth
    st.session_state.openai_client_secret = ""
if 'openai_token_url' not in st.session_state: # For OAuth
    st.session_state.openai_token_url = ""
if 'openai_oauth_token' not in st.session_state: # Stores the fetched token object (can be None)
    st.session_state.openai_oauth_token = None
if 'openai_oauth_token_status' not in st.session_state:
    st.session_state.openai_oauth_token_status = ""

if 'last_openai_auth_signature' not in st.session_state:
    st.session_state.last_openai_auth_signature = ""
if 'available_openai_models' not in st.session_state:
    st.session_state.available_openai_models = []
if 'force_openai_model_refresh' not in st.session_state:
    st.session_state.force_openai_model_refresh = False


# --- Sidebar for Global and Model-specific Settings ---
with st.sidebar:
    st.header(t("general_settings"))
    st.subheader(t("credentials_endpoints"))

    # Ollama Config
    st.session_state.ollama_url = st.text_input(
        t("ollama_base_url"), 
        value=st.session_state.get("ollama_url", "http://localhost:11434"), 
        key="ollama_url_widget"
    )

    # OpenAI-compatible Config Section
    st.markdown(f"#### {t('sidebar_openai_section_header')}")
    
    st.session_state.openai_base_url = st.text_input(
        t("openai_base_url"),
        value=st.session_state.get("openai_base_url", "https://api.openai.com/v1"),
        help=t("openai_base_url_help"),
        key="openai_base_url_widget"
    )

    selected_auth_method_radio = st.radio(
        t("openai_auth_method_label"),
        options=["api_key", "oauth"],
        format_func=lambda x: t("api_key_option") if x == "api_key" else t("oauth_option"),
        key="openai_auth_method_radio_widget", # Changed key to avoid conflict
        horizontal=True,
        index=["api_key", "oauth"].index(st.session_state.openai_auth_method)
    )
    if st.session_state.openai_auth_method != selected_auth_method_radio:
        st.session_state.openai_auth_method = selected_auth_method_radio
        st.session_state.force_openai_model_refresh = True 

    if st.session_state.openai_auth_method == "api_key":
        st.session_state.openai_api_key = st.text_input(
            t("openai_api_key"),
            type="password",
            value=st.session_state.openai_api_key,
            help=t("openai_api_key_help"),
            key="openai_api_key_widget"
        )
    else: # oauth
        st.session_state.openai_client_id = st.text_input(
            t("openai_client_id_label"),
            type="password",
            value=st.session_state.openai_client_id,
            key="openai_client_id_widget"
        )
        st.session_state.openai_client_secret = st.text_input(
            t("openai_client_secret_label"),
            type="password",
            value=st.session_state.openai_client_secret,
            key="openai_client_secret_widget"
        )
        st.session_state.openai_token_url = st.text_input(
            t("openai_token_url_label"),
            value=st.session_state.openai_token_url,
            help=t("openai_token_url_help"),
            key="openai_token_url_widget"
        )
        if st.button(t("get_oauth_token_button"), key="get_oauth_token_btn_widget"):
            if st.session_state.openai_client_id and st.session_state.openai_client_secret and st.session_state.openai_token_url:
                with st.spinner("Obtendo token OAuth..."):
                    token_info, error_msg = get_oauth_token(
                        st.session_state.openai_token_url,
                        st.session_state.openai_client_id,
                        st.session_state.openai_client_secret
                    )
                if token_info:
                    st.session_state.openai_oauth_token = token_info # token_info is a dict or None
                    expires_in_msg = f"(expira em {token_info.get('expires_in', 'N/A')}s)" if token_info.get('expires_in') else ""
                    st.session_state.openai_oauth_token_status = f"{t('oauth_token_success')} {expires_in_msg}"
                    st.session_state.force_openai_model_refresh = True 
                else:
                    st.session_state.openai_oauth_token = None # Explicitly set to None on error
                    st.session_state.openai_oauth_token_status = t("oauth_token_error_fetching").format(error=error_msg)
            else:
                st.session_state.openai_oauth_token_status = t("oauth_token_missing_creds")
        
        st.caption(f"{t('oauth_token_status_label')} {st.session_state.openai_oauth_token_status}")

    # --- Fetch OpenAI Models ---
    current_openai_auth_parts = [st.session_state.get("openai_base_url", "")]
    if st.session_state.openai_auth_method == "api_key":
        current_openai_auth_parts.append(st.session_state.get("openai_api_key", ""))
    else: # oauth
        oauth_token_obj = st.session_state.get("openai_oauth_token") # This will be a dict or None
        if oauth_token_obj and isinstance(oauth_token_obj, dict): # Check if it's a dict
            current_openai_auth_parts.append(oauth_token_obj.get("access_token", ""))
        else:
            current_openai_auth_parts.append("") # Add empty string if no token or not a dict
    
    current_openai_auth_signature = "|".join(filter(None, current_openai_auth_parts))
    
    openai_client_for_models = None
    if st.session_state.openai_base_url and \
       (current_openai_auth_signature != st.session_state.last_openai_auth_signature or \
        st.session_state.force_openai_model_refresh):
        
        st.session_state.force_openai_model_refresh = False 

        if st.session_state.openai_auth_method == "api_key":
            if st.session_state.openai_api_key:
                try:
                    openai_client_for_models = openai.OpenAI(
                        api_key=st.session_state.openai_api_key, 
                        base_url=st.session_state.openai_base_url
                    )
                except Exception as e:
                    st.sidebar.error(f"Erro ao criar cliente (API Key): {str(e)[:100]}...") 
        
        elif st.session_state.openai_auth_method == "oauth":
            token_data_for_client = st.session_state.get("openai_oauth_token") # Again, dict or None
            if token_data_for_client and isinstance(token_data_for_client, dict) and token_data_for_client.get("access_token"):
                try:
                    openai_client_for_models = openai.OpenAI(
                        api_key="DUMMY_VALUE_BEARER_TOKEN_USED", # Can be any non-empty string if not used
                        base_url=st.session_state.openai_base_url,
                        default_headers={"Authorization": f"Bearer {token_data_for_client['access_token']}"}
                    )
                except Exception as e:
                     st.sidebar.error(f"Erro ao criar cliente (OAuth): {str(e)[:100]}...")
            elif not token_data_for_client or not isinstance(token_data_for_client, dict) or not token_data_for_client.get("access_token"):
                # Don't show error here if token just hasn't been fetched yet, list will be empty.
                # Error will show if they try to use it without a token.
                pass


        if openai_client_for_models:
            with st.spinner(t("fetching_models_from").format(base_url=st.session_state.openai_base_url)):
                st.session_state.available_openai_models = get_openai_compatible_models(openai_client_for_models)
            st.session_state.last_openai_auth_signature = current_openai_auth_signature
        else: 
            st.session_state.available_openai_models = []
            if current_openai_auth_signature and current_openai_auth_signature != "|": # Only update if there was some attempt
                 st.session_state.last_openai_auth_signature = current_openai_auth_signature


    # --- Inference Parameters ---
    st.subheader(t("inference_parameters"))
    global_temperature = st.slider(t("temperature"), 0.0, 2.0, 0.7, 0.05, key="global_temp_widget")
    global_max_tokens = st.number_input(t("max_tokens"), 50, 16384, 512, 50, key="global_max_tokens_widget")
    global_top_p = st.slider(t("top_p"), 0.0, 1.0, 0.9, 0.05, key="global_top_p_widget")

    st.markdown("---")
    num_models_to_compare_val = st.radio(
        t("compare_models"), (1, 2, 3), index=0, horizontal=True, key="num_models_radio_widget"
    )
    st.markdown("---")

    model_configs_ui_list = [] # Renamed to avoid confusion
    available_ollama_models = get_ollama_models(st.session_state.ollama_url) if st.session_state.ollama_url else []
    available_openai_models_cached = st.session_state.available_openai_models

    # Ensure model_configs list in session state has the correct number of elements
    # before trying to access them in the loop.
    # This handles changes in num_models_to_compare_val.
    if len(st.session_state.model_configs) != num_models_to_compare_val:
        temp_configs = st.session_state.model_configs[:num_models_to_compare_val]
        while len(temp_configs) < num_models_to_compare_val:
            temp_configs.append({'active': False if len(temp_configs) > 0 else True}) # Default for new models
        st.session_state.model_configs = temp_configs


    for i in range(num_models_to_compare_val):
        st.header(t("model_config_header").format(number=i+1))
        
        # Use a fresh dict for UI construction, then update session state config
        current_ui_config = st.session_state.model_configs[i].copy() if i < len(st.session_state.model_configs) else {'active': (i==0)}


        current_ui_config['active'] = st.checkbox(
            t("activate_model").format(number=i+1), 
            value=current_ui_config.get('active', True if i==0 else False), 
            key=f"active_{i}_widget"
        )
        
        if current_ui_config['active']:
            current_ui_config['endpoint_type'] = st.selectbox(
                t("endpoint_type").format(number=i+1),
                ("Ollama", "OpenAI-compatible"), 
                index=["Ollama", "OpenAI-compatible"].index(current_ui_config.get('endpoint_type', "Ollama")),
                key=f"endpoint_type_{i}_widget"
            )

            if current_ui_config['endpoint_type'] == "Ollama":
                if available_ollama_models:
                    default_ollama_model_idx = 0
                    if current_ui_config.get('model_name') in available_ollama_models:
                        default_ollama_model_idx = available_ollama_models.index(current_ui_config.get('model_name'))
                    elif available_ollama_models: # Ensure list is not empty
                         default_ollama_model_idx = 0 

                    current_ui_config['model_name'] = st.selectbox(
                        t("ollama_model_select").format(number=i+1),
                        options=available_ollama_models,
                        index=default_ollama_model_idx,
                        key=f"ollama_model_{i}_widget"
                    )
                else:
                    current_ui_config['model_name'] = st.text_input(
                        t("ollama_model_text").format(number=i+1),
                        value=current_ui_config.get('model_name', "llama3:latest"),
                        help=t("ollama_model_text_help"),
                        key=f"ollama_model_text_{i}_widget"
                    )
            else: # OpenAI-compatible
                manual_type_openai_key = f"openai_manual_toggle_{i}"
                if manual_type_openai_key not in st.session_state: # Initialize toggle state
                    st.session_state[manual_type_openai_key] = not bool(available_openai_models_cached)

                # Update session state from checkbox immediately
                st.session_state[manual_type_openai_key] = st.checkbox(
                    t("openai_type_manual_toggle").format(number=i+1),
                    value=st.session_state[manual_type_openai_key], # Use existing session state value
                    key=manual_type_openai_key + "_widget"
                )
                
                if not st.session_state[manual_type_openai_key] and available_openai_models_cached:
                    default_openai_model_idx = 0
                    if current_ui_config.get('model_name') in available_openai_models_cached:
                        default_openai_model_idx = available_openai_models_cached.index(current_ui_config.get('model_name'))
                    elif available_openai_models_cached: # Ensure list is not empty
                        default_openai_model_idx = 0

                    current_ui_config['model_name'] = st.selectbox(
                        t("openai_model_select").format(number=i+1),
                        options=available_openai_models_cached,
                        index=default_openai_model_idx,
                        help=t("openai_model_select_help"),
                        key=f"openai_model_select_dd_{i}_widget"
                    )
                else:
                     current_ui_config['model_name'] = st.text_input(
                        t("openai_model_text").format(number=i+1),
                        value=current_ui_config.get('model_name', "gpt-3.5-turbo"),
                        help=t("openai_model_text_alt_help") if not available_openai_models_cached else t("openai_model_text_help"),
                        key=f"openai_model_text_input_{i}_widget"
                    )

            current_ui_config['temperature'] = global_temperature
            current_ui_config['max_tokens'] = global_max_tokens
            current_ui_config['top_p'] = global_top_p
        
        model_configs_ui_list.append(current_ui_config)
        st.markdown("---")
    
    st.session_state.model_configs = model_configs_ui_list # Update session state with all configs from UI


# --- Main Area for Prompt and Responses ---
st.subheader(t("prompt_area_header"))
prompt_text = st.text_area(
    t("prompt_area_label"), 
    height=150, 
    key="prompt_input_widget", 
    value=t("default_prompt_value")
)

if st.button(t("generate_button"), type="primary", key="generate_btn_widget"):
    if not prompt_text.strip():
        st.warning(t("warning_empty_prompt"))
    else:
        st.session_state.response_details = [None] * 3 
        active_configs_with_indices = [
            (idx, conf) for idx, conf in enumerate(st.session_state.model_configs[:num_models_to_compare_val]) 
            if conf.get('active', False)
        ]
        
        if not active_configs_with_indices:
            st.warning(t("info_no_active_models"))
        else:
            with st.spinner(t("spinner_generating")):
                for original_idx, config in active_configs_with_indices:
                    if not config.get('model_name', '').strip():
                        st.session_state.response_details[original_idx] = {
                            "text": t("error_model_not_specified").format(number=original_idx+1),
                            "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}
                        }
                        continue

                    response_data = {}
                    if config['endpoint_type'] == "Ollama":
                        if not st.session_state.ollama_url:
                            response_data = {"text": t("error_ollama_url_not_set").format(number=original_idx+1), "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}}
                        else:
                            response_data = query_ollama(
                                st.session_state.ollama_url, config['model_name'], prompt_text,
                                config['temperature'], config['max_tokens'], config['top_p']
                            )
                    elif config['endpoint_type'] == "OpenAI-compatible":
                        openai_query_client = None
                        error_occurred_creating_client = False

                        if not st.session_state.openai_base_url:
                            response_data = {"text": t("error_openai_base_url_not_set").format(number=original_idx+1), "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}}
                            error_occurred_creating_client = True
                        elif st.session_state.openai_auth_method == "api_key":
                            if not st.session_state.openai_api_key:
                                response_data = {"text": t("error_openai_apikey_not_set").format(number=original_idx+1), "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}}
                                error_occurred_creating_client = True
                            else:
                                try:
                                    openai_query_client = openai.OpenAI(api_key=st.session_state.openai_api_key, base_url=st.session_state.openai_base_url)
                                except Exception as e:
                                    response_data = {"text": t("error_creating_openai_client_api_key").format(number=original_idx+1, e=str(e)), "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}}
                                    error_occurred_creating_client = True
                        elif st.session_state.openai_auth_method == "oauth":
                            token_data_for_query = st.session_state.get("openai_oauth_token")
                            if not token_data_for_query or not isinstance(token_data_for_query, dict) or not token_data_for_query.get("access_token"):
                                response_data = {"text": t("error_openai_oauth_token_not_set").format(number=original_idx+1), "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}}
                                error_occurred_creating_client = True
                            else:
                                try:
                                    openai_query_client = openai.OpenAI(
                                        api_key="DUMMY_VALUE_BEARER_TOKEN_USED", 
                                        base_url=st.session_state.openai_base_url,
                                        default_headers={"Authorization": f"Bearer {token_data_for_query['access_token']}"}
                                    )
                                except Exception as e:
                                    response_data = {"text": t("error_creating_openai_client_oauth").format(number=original_idx+1, e=str(e)), "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}}
                                    error_occurred_creating_client = True
                        
                        if not error_occurred_creating_client and openai_query_client:
                            response_data = query_openai_compatible(
                                openai_query_client, config['model_name'], prompt_text,
                                config['temperature'], config['max_tokens'], config['top_p']
                            )
                        # If error_occurred_creating_client is True, response_data is already set with the error.

                    st.session_state.response_details[original_idx] = response_data
            st.success(t("success_responses_generated"))

# --- Display Responses ---
st.subheader(t("comparison_results_header"))

active_model_indices_with_details = [
    i for i, details in enumerate(st.session_state.response_details[:num_models_to_compare_val])
    if details is not None and i < len(st.session_state.model_configs) and st.session_state.model_configs[i].get('active', False)
]

if active_model_indices_with_details:
    cols = st.columns(len(active_model_indices_with_details))
    
    col_idx = 0
    for i in active_model_indices_with_details:
        config = st.session_state.model_configs[i]
        response_detail = st.session_state.response_details[i]
        
        response_text = response_detail.get('text', t("not_applicable_abbrev"))
        inference_time = response_detail.get('time')
        in_tokens = response_detail.get('in_tokens')
        out_tokens = response_detail.get('out_tokens')

        with cols[col_idx]:
            model_name_from_config = config.get('model_name', '')
            model_display_name = model_name_from_config if model_name_from_config else t("model_not_configured").format(number=i+1)
            
            st.markdown(f"#### {t('model_display_name_header').format(number=i+1, name=model_display_name)}")
            st.markdown(f"**{t('endpoint_label')}** `{config.get('endpoint_type', t('not_applicable_abbrev'))}`")
            
            st.markdown(f"**{t('inference_metrics_header')}**")
            metric_cols = st.columns(3)
            metric_cols[0].metric(
                label=t("inference_time_label"),
                value=f"{inference_time:.2f} {t('seconds_unit_short')}" if inference_time is not None else t("not_applicable_abbrev")
            )
            metric_cols[1].metric(
                label=t("input_tokens_label"),
                value=f"{in_tokens}" if in_tokens is not None else t("not_applicable_abbrev")
            )
            metric_cols[2].metric(
                label=t("output_tokens_label"),
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
                if response_detail.get("raw_response"):
                    try:
                        # If raw_response is already a dict (e.g. from Ollama or error)
                        if isinstance(response_detail["raw_response"], dict):
                            st.json(response_detail["raw_response"], expanded=False)
                        # If raw_response is a JSON string (e.g. from OpenAI pydantic model)
                        elif isinstance(response_detail["raw_response"], str):
                            st.json(json.loads(response_detail["raw_response"]), expanded=False)
                    except json.JSONDecodeError:
                        st.text("Raw response (not valid JSON):")
                        st.text(str(response_detail["raw_response"]))
                    except Exception as e:
                        st.text(f"Error displaying raw response: {e}")


            st.text_area(
                label=t("response_from_model_label").format(model_name=model_display_name),
                value=str(response_text),
                height=350,
                key=f"response_output_{i}_widget",
                disabled=True,
                label_visibility="collapsed"
            )
        col_idx += 1
    
elif any(st.session_state.model_configs[i].get('active') for i in range(num_models_to_compare_val) if i < len(st.session_state.model_configs)):
     st.info(t("info_click_generate"))
else:
    st.info(t("info_no_active_models"))

# --- Usage Tips ---
st.sidebar.markdown("---")
st.sidebar.info(f"""
**{t("how_to_use_header")}**
{t("how_to_use_content")}
""")