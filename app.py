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
        "pt": "Prompt-Playground",
        "en": "Prompt-Playground"
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
        "pt": "Obter/Testar Token OAuth",
        "en": "Get/Test OAuth Token"
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
    "oauth_token_error_auto_fetching": {
        "pt": "Falha ao obter token OAuth automaticamente: {error}",
        "en": "Failed to automatically fetch OAuth token: {error}"
    },
    "error_oauth_token_unavailable": {
        "pt": "Token OAuth não disponível ou inválido. Verifique as credenciais OAuth e tente obter o token na barra lateral, ou verifique os logs se a obtenção automática falhou.",
        "en": "OAuth token not available or invalid. Check OAuth credentials and try fetching token in sidebar, or check logs if automatic fetch failed."
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
        "pt": "Quem foi Patativa do Assaré e qual a sua importância para a cultura do Nordeste do Brasil?",
        "en": "Who was Patativa do Assaré and what is his importance to the culture of Northeastern Brazil?"
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
    "spinner_fetching_oauth_token": {
        "pt": "Obtendo token OAuth...",
        "en": "Fetching OAuth token..."
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
    "error_openai_oauth_creds_not_set_for_auto_fetch": {
        "pt": "Modelo {number} (OpenAI-compatível/OAuth): Client ID, Client Secret ou URL do Token não configurados para obtenção automática de token.",
        "en": "Model {number} (OpenAI-compatible/OAuth): Client ID, Client Secret, or Token URL not configured for automatic token fetching."
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
    *   Se OAuth: Preencha Client ID, Client Secret e URL do Token. O token será obtido automaticamente ao gerar respostas. Use "Obter/Testar Token OAuth" para verificar as credenciais.
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
    *   If OAuth: Fill in Client ID, Client Secret, and Token URL. The token will be fetched automatically when generating responses. Use "Get/Test OAuth Token" to verify credentials.
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
    # Fallback for missing keys during development
    translation_map = translations.get(key, {})
    return translation_map.get(st.session_state.lang, f"[{key}_{st.session_state.lang} - MISSING]")


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
    try:
        client = ollama.Client(host=base_url)
        models_info = client.list()
        return [model['model'] for model in models_info['models']]
    except Exception as e:
        st.sidebar.error(t("error_listing_ollama_models").format(base_url=base_url, e=str(e)))
        return []

def query_ollama(base_url, model_name, prompt, temperature, max_tokens, top_p):
    start_time = time.time()
    try:
        client = ollama.Client(host=base_url)
        response = client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': temperature, 'num_predict': max_tokens, 'top_p': top_p}
        )
        end_time = time.time()
        return {
            "text": response['message']['content'], "time": end_time - start_time,
            "in_tokens": response.get('prompt_eval_count'), "out_tokens": response.get('eval_count'),
            "raw_response": response
        }
    except Exception as e:
        return {
            "text": t("error_querying_ollama").format(model_name=model_name, e=str(e)),
            "time": time.time() - start_time, "in_tokens": None, "out_tokens": None,
            "raw_response": {"error": str(e)}
        }

def get_oauth_token(token_url, client_id, client_secret):
    try:
        response = requests.post(
            token_url,
            data={'grant_type': 'client_credentials', 'client_id': client_id, 'client_secret': client_secret},
            headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'},
            timeout=15
        )
        response.raise_for_status()
        token_data = response.json()
        if "access_token" not in token_data:
            return None, f"OAuth token response did not contain 'access_token'. Received: {str(token_data)[:200]}"
        return token_data, None
    except requests.exceptions.HTTPError as e:
        err_detail = str(e)
        if e.response is not None:
            try: err_detail += f" - Response: {e.response.json()}"
            except json.JSONDecodeError: err_detail += f" - Response: {e.response.text[:200]}"
        return None, f"HTTPError: {err_detail}"
    except requests.exceptions.RequestException as e: return None, f"RequestException: {str(e)}"
    except json.JSONDecodeError as e:
        resp_text = response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'
        return None, f"JSONDecodeError: {str(e)}. Response text: {resp_text[:200]}"
    except Exception as e: return None, f"Generic error: {str(e)}"

def get_openai_compatible_models(client: openai.OpenAI):
    try:
        return [model.id for model in client.models.list().data]
    except Exception: return []

def query_openai_compatible(client: openai.OpenAI, model_name: str, prompt: str, temperature: float, max_tokens: int, top_p: float):
    start_time = time.time()
    auth_method = st.session_state.get("openai_auth_method", "api_key")
    msg = ""
    try:
        completion = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, top_p=top_p
        )
        return {
            "text": completion.choices[0].message.content, "time": time.time() - start_time,
            "in_tokens": getattr(completion.usage, 'prompt_tokens', None),
            "out_tokens": getattr(completion.usage, 'completion_tokens', None),
            "raw_response": completion.model_dump_json(indent=2)
        }
    except openai.APIConnectionError as e: msg = t("error_openai_connection").format(model_name=model_name, e=str(e))
    except openai.AuthenticationError as e:
        msg = t("error_openai_auth_oauth" if auth_method == "oauth" else "error_openai_auth_apikey").format(model_name=model_name, e=str(e))
    except openai.RateLimitError as e: msg = t("error_openai_ratelimit").format(model_name=model_name, e=str(e))
    except openai.NotFoundError as e:
        base_url_str = str(client.base_url) if client else t("not_applicable_abbrev")
        msg = t("error_openai_notfound").format(model_name=model_name, base_url=base_url_str, e=str(e))
    except Exception as e:
        try:
            err_body = getattr(e, 'response', None)
            err_detail = err_body.json().get("error", {}).get("message", str(e)) if err_body else str(e)
            msg = t("error_querying_openai_api").format(model_name=model_name, msg=err_detail)
        except (json.JSONDecodeError, AttributeError): msg = t("error_querying_openai_api").format(model_name=model_name, msg=str(e))
    
    return {"text": msg, "time": time.time() - start_time, "in_tokens": None, "out_tokens": None, "raw_response": {"error": msg, "details": str(e) if 'e' in locals() else "Unknown"}}

# --- Streamlit Interface ---
st.title(t("app_title"))
st.markdown(t("app_subtitle"))

# --- Session State Initialization ---
default_ss = {
    'response_details': [None] * 3, 'model_configs': [{}, {}, {}],
    'openai_auth_method': "api_key", 'openai_api_key': "",
    'openai_client_id': "", 'openai_client_secret': "", 'openai_token_url': "",
    'openai_oauth_token': None, 'openai_oauth_token_status': "",
    'last_openai_auth_signature': "", 'available_openai_models': [],
    'force_openai_model_refresh': False, 'ollama_url': "http://localhost:11434",
    'openai_base_url': "https://api.openai.com/v1",
    'global_temp': 0.7, 'global_max_tokens': 1024, 'global_top_p': 0.9,
    'num_models': 1, 'prompt_input': t("default_prompt_value")
}
for k, v in default_ss.items():
    if k not in st.session_state: st.session_state[k] = v
if len(st.session_state.model_configs) < 3: # Ensure list has 3 elements
    st.session_state.model_configs.extend([{}] * (3 - len(st.session_state.model_configs)))


# --- Sidebar for Global and Model-specific Settings ---
with st.sidebar:
    st.header(t("general_settings"))
    st.subheader(t("credentials_endpoints"))

    st.session_state.ollama_url = st.text_input(t("ollama_base_url"), value=st.session_state.ollama_url, key="ollama_url_widget")

    st.markdown(f"#### {t('sidebar_openai_section_header')}")
    st.session_state.openai_base_url = st.text_input(t("openai_base_url"), value=st.session_state.openai_base_url, help=t("openai_base_url_help"), key="openai_base_url_widget")

    selected_auth_method_radio = st.radio(
        t("openai_auth_method_label"), ["api_key", "oauth"], horizontal=True,
        format_func=lambda x: t("api_key_option") if x == "api_key" else t("oauth_option"),
        index=["api_key", "oauth"].index(st.session_state.openai_auth_method),
        key="openai_auth_method_radio_widget"
    )
    if st.session_state.openai_auth_method != selected_auth_method_radio:
        st.session_state.openai_auth_method = selected_auth_method_radio
        st.session_state.force_openai_model_refresh = True

    if st.session_state.openai_auth_method == "api_key":
        st.session_state.openai_api_key = st.text_input(t("openai_api_key"), type="password", value=st.session_state.openai_api_key, help=t("openai_api_key_help"), key="openai_api_key_widget")
    else: # oauth
        st.session_state.openai_client_id = st.text_input(t("openai_client_id_label"), value=st.session_state.openai_client_id, key="openai_client_id_widget")
        st.session_state.openai_client_secret = st.text_input(t("openai_client_secret_label"), type="password", value=st.session_state.openai_client_secret, key="openai_client_secret_widget")
        st.session_state.openai_token_url = st.text_input(t("openai_token_url_label"), value=st.session_state.openai_token_url, help=t("openai_token_url_help"), key="openai_token_url_widget")
        
        if st.button(t("get_oauth_token_button"), key="get_oauth_token_btn_widget_sidebar"):
            if st.session_state.openai_client_id and st.session_state.openai_client_secret and st.session_state.openai_token_url:
                with st.spinner(t("spinner_fetching_oauth_token")):
                    token_info, error_msg = get_oauth_token(st.session_state.openai_token_url, st.session_state.openai_client_id, st.session_state.openai_client_secret)
                if token_info:
                    st.session_state.openai_oauth_token = token_info
                    expires_msg = f"(expira em {token_info.get('expires_in', 'N/A')}s)" if token_info.get('expires_in') else ""
                    st.session_state.openai_oauth_token_status = f"{t('oauth_token_success')} {expires_msg}"
                    st.session_state.force_openai_model_refresh = True
                else:
                    st.session_state.openai_oauth_token = None
                    st.session_state.openai_oauth_token_status = t("oauth_token_error_fetching").format(error=error_msg)
            else:
                st.session_state.openai_oauth_token_status = t("oauth_token_missing_creds")
        st.caption(f"{t('oauth_token_status_label')} {st.session_state.openai_oauth_token_status}")

    # --- Fetch OpenAI Models (uses token from session_state, possibly set by button or last auto-fetch) ---
    current_openai_auth_parts = [st.session_state.openai_base_url]
    if st.session_state.openai_auth_method == "api_key":
        current_openai_auth_parts.append(st.session_state.openai_api_key)
    else:
        oauth_token_obj = st.session_state.openai_oauth_token
        if oauth_token_obj and isinstance(oauth_token_obj, dict):
            current_openai_auth_parts.append(oauth_token_obj.get("access_token", ""))
    current_openai_auth_signature = "|".join(filter(None, current_openai_auth_parts))

    if st.session_state.openai_base_url and (current_openai_auth_signature != st.session_state.last_openai_auth_signature or st.session_state.force_openai_model_refresh):
        st.session_state.force_openai_model_refresh = False
        openai_client_for_models = None
        if st.session_state.openai_auth_method == "api_key" and st.session_state.openai_api_key:
            try: openai_client_for_models = openai.OpenAI(api_key=st.session_state.openai_api_key, base_url=st.session_state.openai_base_url)
            except Exception as e: st.sidebar.error(f"Client Error (API Key): {str(e)[:100]}...")
        elif st.session_state.openai_auth_method == "oauth":
            token_data = st.session_state.openai_oauth_token
            if token_data and isinstance(token_data, dict) and token_data.get("access_token"):
                try: openai_client_for_models = openai.OpenAI(api_key="NA_OAUTH_TOKEN_USED", base_url=st.session_state.openai_base_url, default_headers={"Authorization": f"Bearer {token_data['access_token']}"})
                except Exception as e: st.sidebar.error(f"Client Error (OAuth): {str(e)[:100]}...")
        
        if openai_client_for_models:
            with st.spinner(t("fetching_models_from").format(base_url=st.session_state.openai_base_url)):
                st.session_state.available_openai_models = get_openai_compatible_models(openai_client_for_models)
        else: st.session_state.available_openai_models = []
        st.session_state.last_openai_auth_signature = current_openai_auth_signature

    st.subheader(t("inference_parameters"))
    st.session_state.global_temp = st.slider(t("temperature"), 0.0, 2.0, st.session_state.global_temp, 0.05, key="global_temp_widget")
    st.session_state.global_max_tokens = st.number_input(t("max_tokens"), 50, 16384, st.session_state.global_max_tokens, 50, key="global_max_tokens_widget")
    st.session_state.global_top_p = st.slider(t("top_p"), 0.0, 1.0, st.session_state.global_top_p, 0.05, key="global_top_p_widget")

    st.markdown("---")
    st.session_state.num_models = st.radio(t("compare_models"), (1, 2, 3), index=st.session_state.num_models -1, horizontal=True, key="num_models_radio_widget")
    st.markdown("---")

    model_configs_ui_list = []
    available_ollama_models = get_ollama_models(st.session_state.ollama_url) if st.session_state.ollama_url else []
    
    # Ensure model_configs list in session state has the correct number of elements
    if len(st.session_state.model_configs) != st.session_state.num_models:
        temp_configs = st.session_state.model_configs[:st.session_state.num_models]
        while len(temp_configs) < st.session_state.num_models:
            temp_configs.append({'active': (len(temp_configs) == 0)}) # First new model active
        st.session_state.model_configs = temp_configs

    for i in range(st.session_state.num_models):
        st.header(t("model_config_header").format(number=i+1))
        current_ui_config = st.session_state.model_configs[i].copy()

        # current_ui_config['active'] = st.checkbox(t("activate_model").format(number=i+1), value=current_ui_config.get('active', i==0), key=f"active_{i}_widget")
        current_ui_config['active'] = st.checkbox(t("activate_model").format(number=i+1), value=1, key=f"active_{i}_widget")
        
        if current_ui_config['active']:
            current_ui_config['endpoint_type'] = st.selectbox(t("endpoint_type").format(number=i+1), ("Ollama", "OpenAI-compatible"), index=["Ollama", "OpenAI-compatible"].index(current_ui_config.get('endpoint_type', "Ollama")), key=f"endpoint_type_{i}_widget")

            if current_ui_config['endpoint_type'] == "Ollama":
                idx = 0
                if available_ollama_models:
                    if current_ui_config.get('model_name') in available_ollama_models: idx = available_ollama_models.index(current_ui_config.get('model_name'))
                    current_ui_config['model_name'] = st.selectbox(t("ollama_model_select").format(number=i+1), available_ollama_models, index=idx, key=f"ollama_model_{i}_widget")
                else:
                    current_ui_config['model_name'] = st.text_input(t("ollama_model_text").format(number=i+1), value=current_ui_config.get('model_name', "llama3:latest"), help=t("ollama_model_text_help"), key=f"ollama_model_text_{i}_widget")
            else: # OpenAI-compatible
                manual_key = f"openai_manual_toggle_{i}"
                if manual_key not in st.session_state: st.session_state[manual_key] = not bool(st.session_state.available_openai_models)
                st.session_state[manual_key] = st.checkbox(t("openai_type_manual_toggle").format(number=i+1), value=st.session_state[manual_key], key=manual_key + "_widget")
                
                idx = 0
                if not st.session_state[manual_key] and st.session_state.available_openai_models:
                    if current_ui_config.get('model_name') in st.session_state.available_openai_models: idx = st.session_state.available_openai_models.index(current_ui_config.get('model_name'))
                    current_ui_config['model_name'] = st.selectbox(t("openai_model_select").format(number=i+1), st.session_state.available_openai_models, index=idx, help=t("openai_model_select_help"), key=f"openai_model_select_dd_{i}_widget")
                else:
                    current_ui_config['model_name'] = st.text_input(t("openai_model_text").format(number=i+1), value=current_ui_config.get('model_name', "gpt-3.5-turbo"), help=t("openai_model_text_alt_help") if not st.session_state.available_openai_models else t("openai_model_text_help"), key=f"openai_model_text_input_{i}_widget")
            
            current_ui_config['temperature'] = st.session_state.global_temp
            current_ui_config['max_tokens'] = st.session_state.global_max_tokens
            current_ui_config['top_p'] = st.session_state.global_top_p
        
        st.session_state.model_configs[i] = current_ui_config # Update list in place
        st.markdown("---")

# --- Main Area for Prompt and Responses ---
st.subheader(t("prompt_area_header"))
st.session_state.prompt_input = st.text_area(t("prompt_area_label"), height=150, key="prompt_input_main_widget", value=st.session_state.prompt_input)

if st.button(t("generate_button"), type="primary", key="generate_btn_widget"):
    if not st.session_state.prompt_input.strip():
        st.warning(t("warning_empty_prompt"))
    else:
        st.session_state.response_details = [None] * 3
        active_configs = [(idx, conf) for idx, conf in enumerate(st.session_state.model_configs[:st.session_state.num_models]) if conf.get('active')]
        
        if not active_configs:
            st.warning(t("info_no_active_models"))
        else:
            # --- Auto-fetch OAuth token if needed for this run ---
            globally_fetched_oauth_token_this_run = None
            globally_fetched_oauth_error_this_run = None
            needs_oauth_for_this_run = False

            if st.session_state.openai_auth_method == "oauth":
                for _, conf in active_configs:
                    if conf.get('endpoint_type') == "OpenAI-compatible":
                        needs_oauth_for_this_run = True
                        break
            
            if needs_oauth_for_this_run:
                if not (st.session_state.openai_client_id and st.session_state.openai_client_secret and st.session_state.openai_token_url):
                    globally_fetched_oauth_error_this_run = t("oauth_token_missing_creds")
                    st.session_state.openai_oauth_token_status = globally_fetched_oauth_error_this_run # Update sidebar status
                else:
                    with st.spinner(t("spinner_fetching_oauth_token")):
                        token_info, error_msg = get_oauth_token(st.session_state.openai_token_url, st.session_state.openai_client_id, st.session_state.openai_client_secret)
                    if token_info:
                        globally_fetched_oauth_token_this_run = token_info
                        st.session_state.openai_oauth_token = token_info # Update global state
                        expires_msg = f"(expira em {token_info.get('expires_in', 'N/A')}s)" if token_info.get('expires_in') else ""
                        st.session_state.openai_oauth_token_status = f"{t('oauth_token_success')} {expires_msg}"
                        st.session_state.force_openai_model_refresh = True # In case model list needs update
                    else:
                        globally_fetched_oauth_error_this_run = error_msg
                        st.session_state.openai_oauth_token = None # Clear global token on error
                        st.session_state.openai_oauth_token_status = t("oauth_token_error_fetching").format(error=error_msg)
                        st.session_state.force_openai_model_refresh = True

            # --- Process models ---
            with st.spinner(t("spinner_generating")):
                for original_idx, config in active_configs:
                    if not config.get('model_name', '').strip():
                        st.session_state.response_details[original_idx] = {"text": t("error_model_not_specified").format(number=original_idx+1), "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}}
                        continue

                    response_data = {}
                    if config['endpoint_type'] == "Ollama":
                        if not st.session_state.ollama_url: response_data = {"text": t("error_ollama_url_not_set").format(number=original_idx+1), "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}}
                        else: response_data = query_ollama(st.session_state.ollama_url, config['model_name'], st.session_state.prompt_input, config['temperature'], config['max_tokens'], config['top_p'])
                    
                    elif config['endpoint_type'] == "OpenAI-compatible":
                        client = None
                        error_msg_client = ""

                        if not st.session_state.openai_base_url: error_msg_client = t("error_openai_base_url_not_set").format(number=original_idx+1)
                        elif st.session_state.openai_auth_method == "api_key":
                            if not st.session_state.openai_api_key: error_msg_client = t("error_openai_apikey_not_set").format(number=original_idx+1)
                            else:
                                try: client = openai.OpenAI(api_key=st.session_state.openai_api_key, base_url=st.session_state.openai_base_url)
                                except Exception as e: error_msg_client = t("error_creating_openai_client_api_key").format(number=original_idx+1, e=str(e))
                        
                        elif st.session_state.openai_auth_method == "oauth":
                            if globally_fetched_oauth_error_this_run: # Check error from global fetch first
                                if globally_fetched_oauth_error_this_run == t("oauth_token_missing_creds"):
                                     error_msg_client = t("error_openai_oauth_creds_not_set_for_auto_fetch").format(number=original_idx+1)
                                else:
                                     error_msg_client = t("oauth_token_error_auto_fetching").format(error=globally_fetched_oauth_error_this_run)
                            elif not globally_fetched_oauth_token_this_run or not globally_fetched_oauth_token_this_run.get("access_token"):
                                error_msg_client = t("error_oauth_token_unavailable").format(number=original_idx+1) # Should not happen if logic above is correct
                            else:
                                try: client = openai.OpenAI(api_key="NA_OAUTH_TOKEN_USED", base_url=st.session_state.openai_base_url, default_headers={"Authorization": f"Bearer {globally_fetched_oauth_token_this_run['access_token']}"})
                                except Exception as e: error_msg_client = t("error_creating_openai_client_oauth").format(number=original_idx+1, e=str(e))

                        if error_msg_client: response_data = {"text": error_msg_client, "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}}
                        elif client: response_data = query_openai_compatible(client, config['model_name'], st.session_state.prompt_input, config['temperature'], config['max_tokens'], config['top_p'])
                        else: response_data = {"text": "Cliente OpenAI não pôde ser inicializado (erro desconhecido).", "time": 0, "in_tokens": None, "out_tokens": None, "raw_response": {}}


                    st.session_state.response_details[original_idx] = response_data
            st.success(t("success_responses_generated"))
            if needs_oauth_for_this_run and st.session_state.force_openai_model_refresh : # If token fetch triggered refresh, rerun
                st.rerun()


# --- Display Responses ---
st.subheader(t("comparison_results_header"))
active_indices_with_details = [
    i for i, details in enumerate(st.session_state.response_details[:st.session_state.num_models])
    if details is not None and i < len(st.session_state.model_configs) and st.session_state.model_configs[i].get('active', False)
]

if active_indices_with_details:
    cols = st.columns(len(active_indices_with_details))
    for col_idx, i in enumerate(active_indices_with_details):
        conf = st.session_state.model_configs[i]
        resp_detail = st.session_state.response_details[i]
        
        name = conf.get('model_name', '') or t("model_not_configured").format(number=i+1)
        
        with cols[col_idx]:
            st.markdown(f"#### {t('model_display_name_header').format(number=i+1, name=name)}")
            st.markdown(f"**{t('endpoint_label')}** `{conf.get('endpoint_type', t('not_applicable_abbrev'))}`")
            
            st.markdown(f"**{t('inference_metrics_header')}**")
            m_cols = st.columns(3)
            m_cols[0].metric(t("inference_time_label"), f"{resp_detail.get('time', 0):.2f} {t('seconds_unit_short')}" if resp_detail.get('time') is not None else t("not_applicable_abbrev"))
            m_cols[1].metric(t("input_tokens_label"), str(resp_detail.get('in_tokens', 'N/A')))
            m_cols[2].metric(t("output_tokens_label"), str(resp_detail.get('out_tokens', 'N/A')))

            with st.expander(t("parameters_used_expander")):
                st.caption(t("parameters_caption").format(temp=conf.get('temperature', st.session_state.global_temp), max_tokens=conf.get('max_tokens', st.session_state.global_max_tokens), top_p=conf.get('top_p', st.session_state.global_top_p)))
                if resp_detail.get("raw_response"):
                    try:
                        raw = resp_detail["raw_response"]
                        st.json(json.loads(raw) if isinstance(raw, str) else raw, expanded=False)
                    except Exception: st.text(str(raw)[:1000] + "...")
            
            st.text_area(t("response_from_model_label").format(model_name=name), str(resp_detail.get('text', '')), height=350, key=f"response_output_{i}_widget", disabled=True, label_visibility="collapsed")

elif any(st.session_state.model_configs[i].get('active') for i in range(st.session_state.num_models) if i < len(st.session_state.model_configs)):
    st.info(t("info_click_generate"))
else:
    st.info(t("info_no_active_models"))

st.sidebar.markdown("---")
st.sidebar.info(f"**{t('how_to_use_header')}**\n{t('how_to_use_content')}")