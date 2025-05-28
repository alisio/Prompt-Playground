import streamlit as st
import openai
import json
import time
import requests
import uuid # For unique keys for models

# --- Language Configuration (updated) ---
translations = {
    "page_title": {"pt": "TestaAI", "en": "TestaAI"},
    "app_header_title": {"pt": "🧪 TestaAI", "en": "🧪 TestaAI"},
    "app_subtitle": {"pt": "Seu Playground para Experimentos com LLMs", "en": "Your Playground for LLM Experiments"},

    "models_active_counter": {"pt": "Modelo(s) Ativo(s)", "en": "Active Model(s)"},
    "comparisons_run_counter": {"pt": "Comparações", "en": "Comparisons"},

    "sidebar_configurations_header": {"pt": "⚙️ Configurações", "en": "⚙️ Configurations"},
    "language": {"pt": "Idioma", "en": "Language"},
    "ai_models_header": {"pt": "Modelos de IA (OpenAI-compatível)", "en": "AI Models (OpenAI-compatible)"},
    "add_model_button": {"pt": "➕ Adicionar Modelo", "en": "➕ Add Model"},
    "max_models_reached": {"pt": "Máximo de {max} modelos atingido.", "en": "Maximum of {max} models reached."},
    "model_user_given_name_label": {"pt": "Nome do Modelo (Apelido)", "en": "Model Name (Nickname)"}, # Label for input
    "activate_model_toggle_help": {"pt": "Ativar/Desativar este modelo", "en": "Activate/Deactivate this model"}, # Help for toggle
    "remove_model_button_help": {"pt": "Remover este modelo", "en": "Remove this model"}, # Help for button
    "model_advanced_settings_expander": {"pt": "Configurações de \"{name}\"", "en": "Settings for \"{name}\""}, # New

    "service_url_label": {"pt": "URL Base da API", "en": "API Base URL"},
    "service_url_select_option_other": {"pt": "Outro (digitar)", "en": "Other (type)"},
    "service_url_custom_label": {"pt": "URL Base da API (Personalizada)", "en": "API Base URL (Custom)"},
    "service_url_help_openai": {"pt": "Ex: https://api.openai.com/v1 ou URL do seu endpoint privado", "en": "Ex: https://api.openai.com/v1 or your private endpoint URL"},
    "auth_method_label": {"pt": "Método de Autenticação", "en": "Authentication Method"},
    "api_key_option": {"pt": "Chave de API", "en": "API Key"},
    "oauth_option": {"pt": "OAuth (Client Credentials)", "en": "OAuth (Client Credentials)"},
    "api_key_label": {"pt": "Chave API", "en": "API Key"},
    "api_key_help": {"pt": "Pode ser 'NA' se o endpoint não exigir chave e o método for Chave API.", "en": "Can be 'NA' if the endpoint doesn't require a key and method is API Key."},
    "client_id_label": {"pt": "Client ID (OAuth)", "en": "Client ID (OAuth)"},
    "client_secret_label": {"pt": "Client Secret (OAuth)", "en": "Client Secret (OAuth)"},
    "token_url_label": {"pt": "URL do Token (OAuth)", "en": "Token URL (OAuth)"},
    "token_url_help": {"pt": "Ex: https://seu-servidor-auth.com/oauth/token", "en": "Ex: https://your-auth-server.com/oauth/token"},
    "get_oauth_token_button": {"pt": "Obter/Testar Token OAuth", "en": "Get/Test OAuth Token"},
    "oauth_token_status_label": {"pt": "Status do Token OAuth:", "en": "OAuth Token Status:"},
    "oauth_token_success": {"pt": "Token OAuth obtido com sucesso!", "en": "OAuth Token obtained successfully!"},
    "oauth_token_missing_creds": {"pt": "Preencha Client ID, Client Secret e URL do Token.", "en": "Please fill in Client ID, Client Secret, and Token URL."},
    "oauth_token_error_fetching": {"pt": "Erro ao obter token OAuth: {error}", "en": "Error fetching OAuth token: {error}"},
    "model_identifier_label_openai": {"pt": "Nome do Modelo na API", "en": "API Model Name"},
    "model_identifier_help_select": {"pt": "Selecione ou digite o nome do modelo se não estiver na lista.", "en": "Select or type the model name if not in the list."},
    "model_identifier_help_text_openai": {"pt": "Ex: gpt-4, gpt-3.5-turbo. Modelos não puderam ser listados ou digitação manual ativa.", "en": "Ex: gpt-4, gpt-3.5-turbo. Models could not be listed or manual input active."},
    "manual_model_name_toggle": {"pt": "Digitar nome do modelo da API manualmente", "en": "Manually type API model name"},
    "temperature_label": {"pt": "Temperatura", "en": "Temperature"},
    "max_tokens_label": {"pt": "Max Tokens", "en": "Max Tokens"},
    "top_p_label": {"pt": "Top P", "en": "Top P"},
    "prompt_area_header": {"pt": "⚡ Enviar Prompt para {count} modelo(s)", "en": "⚡ Send Prompt to {count} model(s)"},
    "prompt_area_label": {"pt": "Digite seu prompt aqui... (Ctrl+Enter para enviar)", "en": "Enter your prompt here... (Ctrl+Enter to send)"},
    "prompt_char_count_label": {"pt": "Caracteres: {count}", "en": "Characters: {count}"},
    "default_prompt_value": {"pt": "Quem foi Patativa do Assaré e qual é a sua importância para a cultura do Nordeste brasileiro?", "en": "Who was Patativa do Assaré and what is his importance to the culture of Brazil's Northeast?"},
    "send_button": {"pt": "✉️ Enviar", "en": "✉️ Send"},
    "warning_empty_prompt": {"pt": "⚠️ Por favor, insira um prompt.", "en": "⚠️ Please enter a prompt."},
    "warning_no_active_models_configured": {"pt": "⚠️ Nenhum modelo ativo e configurado. Configure e ative modelos na barra lateral.", "en": "⚠️ No active and configured models. Configure and activate models in the sidebar."},
    "spinner_generating": {"pt": "Gerando respostas...", "en": "Generating responses..."},
    "spinner_fetching_oauth_token": {"pt": "Obtendo token OAuth...", "en": "Fetching OAuth token..."},
    "error_model_not_specified": {"pt": "{model_name}: Nome do modelo na API não especificado.", "en": "{model_name}: API Model name not specified."},
    "error_service_url_not_set": {"pt": "{model_name}: URL Base da API não configurada ou inválida.", "en": "{model_name}: API Base URL not configured or invalid."},
    "error_api_key_not_set": {"pt": "{model_name} (API Key): Chave API não configurada.", "en": "{model_name} (API Key): API Key not configured."},
    "error_oauth_creds_not_set_for_auto_fetch": {"pt": "{model_name} (OAuth): Client ID, Client Secret ou URL do Token não configurados.", "en": "{model_name} (OAuth): Client ID, Client Secret, or Token URL not configured."},
    "error_oauth_token_unavailable": {"pt": "{model_name} (OAuth): Token OAuth não disponível ou inválido.", "en": "{model_name} (OAuth): OAuth token not available or invalid."},
    "error_creating_openai_client_api_key": {"pt": "{model_name} (API Key): Erro ao criar cliente OpenAI: {e}", "en": "{model_name} (API Key): Error creating OpenAI client: {e}"},
    "error_creating_openai_client_oauth": {"pt": "{model_name} (OAuth): Erro ao criar cliente OpenAI: {e}", "en": "{model_name} (OAuth): Error creating OpenAI client: {e}"},
    "success_responses_generated": {"pt": "✅ Respostas geradas!", "en": "✅ Responses generated!"},
    "comparison_results_header": {"pt": "Resultados da Comparação", "en": "Comparison Results"},
    "model_display_name_header": {"pt": "{user_name}", "en": "{user_name}"},
    "model_tech_name_subheader": {"pt": "Modelo API: {tech_name}", "en": "API Model: {tech_name}"},
    "not_applicable_abbrev": {"pt": "N/A", "en": "N/A"},
    "parameters_used_expander": {"pt": "Parâmetros Usados", "en": "Parameters Used"},
    "parameters_caption": {"pt": "Temp: {temp}, Max Tokens: {max_tokens}, Top P: {top_p}", "en": "Temp: {temp}, Max Tokens: {max_tokens}, Top P: {top_p}"},
    "response_from_model_label": {"pt": "Resposta:", "en": "Response:"},
    "info_placeholder_no_results": {"pt": "⚡ Pronto para comparar modelos\nConfigure os modelos e envie um prompt para começar", "en": "⚡ Ready to compare models\nConfigure the models and send a prompt to get started"},
    "info_click_send": {"pt": "Clique em 'Enviar' após configurar os modelos e inserir um prompt.", "en": "Click 'Send' after configuring models and entering a prompt."},
    "fetching_models_from": {"pt": "Buscando modelos de {base_url}...", "en": "Fetching models from {base_url}..."},
    "error_openai_connection": {"pt": "Erro de conexão com API ({model_name}): {e}", "en": "API connection error ({model_name}): {e}"},
    "error_openai_auth_apikey": {"pt": "Erro de autenticação API ({model_name}): Chave inválida/faltando? {e}", "en": "API authentication error ({model_name}): Invalid/missing key? {e}"},
    "error_openai_auth_oauth": {"pt": "Erro de autenticação API ({model_name}) via OAuth: Token inválido/expirado? {e}", "en": "API authentication error ({model_name}) via OAuth: Invalid/expired token? {e}"},
    "error_openai_ratelimit": {"pt": "Erro de limite de taxa API ({model_name}): {e}", "en": "API rate limit error ({model_name}): {e}"},
    "error_openai_notfound": {"pt": "Erro: Modelo '{model_id}' não encontrado em '{base_url}'. {e}", "en": "Error: Model '{model_id}' not found at '{base_url}'. {e}"},
    "error_querying_openai_api": {"pt": "Erro ao consultar API ({model_name}): {msg}", "en": "Error querying API ({model_name}): {msg}"},
    "inference_time_label": {"pt": "Tempo", "en": "Time"},
    "input_tokens_label": {"pt": "Entrada", "en": "Input"},
    "output_tokens_label": {"pt": "Saída", "en": "Output"},
    "tokens_unit": {"pt": "tokens", "en": "tokens"},
    "seconds_unit_short": {"pt": "s", "en": "s"},
    "inference_metrics_header": {"pt": "Métricas", "en": "Metrics"},
    "error_general_response_area": {"pt": "Erro: {msg}", "en": "Error: {msg}"},
    "nothing_to_display": {"pt": "Nada para exibir.", "en": "Nothing to display."},
    "prompt_placeholder_no_active_model": {"pt": "⚠️ Nenhum modelo ativo", "en": "⚠️ No active model"},

    # New translations for added sections
    "how_to_operate_header": {"pt": "📖 Como Operar o Dashboard", "en": "📖 How to Operate the Dashboard"},
    "how_to_operate_intro": {
        "pt": "Siga estes passos para comparar LLMs:",
        "en": "Follow these steps to compare LLMs:"
    },
    "how_to_operate_step1_config": {
        "pt": "1. **Configurar Modelos (na barra lateral):**",
        "en": "1. **Configure Models (in the sidebar):**"
    },
    "how_to_operate_step1_detail_add": {
        "pt": "   - Clique em '➕ Adicionar Modelo' para adicionar um novo slot de modelo (até {max_models}).",
        "en": "   - Click '➕ Add Model' to add a new model slot (up to {max_models})."
    },
    "how_to_operate_step1_detail_nickname": {
        "pt": "   - Dê um **Nome (Apelido)** para fácil identificação.",
        "en": "   - Provide a **Model Name (Nickname)** for easy identification."
    },
    "how_to_operate_step1_detail_url": {
        "pt": "   - Expanda as 'Configurações' do modelo. Selecione ou insira a **URL Base da API** (ex: `https://api.openai.com/v1` para OpenAI, ou seu endpoint local como `http://localhost:11434/v1` para Ollama/LM Studio).",
        "en": "   - Expand the model's 'Settings'. Select or enter the **API Base URL** (e.g., `https://api.openai.com/v1` for OpenAI, or your local endpoint like `http://localhost:11434/v1` for Ollama/LM Studio)."
    },
    "how_to_operate_step1_detail_auth": {
        "pt": "   - Escolha o **Método de Autenticação**:",
        "en": "   - Choose the **Authentication Method**:"
    },
    "how_to_operate_step1_detail_auth_apikey": {
        "pt": "     - **Chave de API:** Insira sua chave. Para alguns endpoints locais (como Ollama), pode ser 'NA' ou qualquer valor se não for necessária.",
        "en": "     - **API Key:** Enter your key. For some local endpoints (like Ollama), it can be 'NA' or any value if not required."
    },
    "how_to_operate_step1_detail_auth_oauth": {
        "pt": "     - **OAuth:** Preencha Client ID, Client Secret, URL do Token e clique em 'Obter/Testar Token OAuth'.",
        "en": "     - **OAuth:** Fill in Client ID, Client Secret, Token URL, and click 'Get/Test OAuth Token'."
    },
    "how_to_operate_step1_detail_model_api": {
        "pt": "   - Especifique o **Nome do Modelo na API** (ex: `gpt-4`, `llama3`). Se a lista de modelos for carregada, você pode selecionar um. Caso contrário, ative 'Digitar nome do modelo da API manualmente' e insira.",
        "en": "   - Specify the **API Model Name** (e.g., `gpt-4`, `llama3`). If the model list loads, you can select one. Otherwise, enable 'Manually type API model name' and enter it."
    },
    "how_to_operate_step1_detail_params": {
        "pt": "   - Ajuste **Temperatura**, **Max Tokens** e **Top P** conforme necessário.",
        "en": "   - Adjust **Temperature**, **Max Tokens**, and **Top P** as needed."
    },
    "how_to_operate_step2_activate": {
        "pt": "2. **Ativar Modelos:** Use o botão de alternância (toggle ao lado do nome do modelo) para ativá-lo para comparação. Modelos ativos são indicados por '✅'.",
        "en": "2. **Activate Models:** Use the toggle switch (next to the model name) to activate it for comparison. Active models are indicated by '✅'."
    },
    "how_to_operate_step3_prompt": {
        "pt": "3. **Enviar Prompt:** Digite seu prompt na área principal e clique em '✉️ Enviar'.",
        "en": "3. **Send Prompt:** Enter your prompt in the main area and click '✉️ Send'."
    },
    "how_to_operate_step4_results": {
        "pt": "4. **Analisar Resultados:** As respostas de cada modelo ativo aparecerão lado a lado para comparação, incluindo métricas como tempo de inferência e uso de tokens.",
        "en": "4. **Analyze Results:** Responses from each active model will appear side-by-side for comparison, including metrics like inference time and token usage."
    },
    "source_code_header": {"pt": "📦 Código Fonte", "en": "📦 Source Code"},
    "source_code_text": {
        "pt": "O código fonte desta aplicação está disponível no GitHub:",
        "en": "The source code for this application is available on GitHub:"
    },
    "author_section_header": {"pt": "👤 Sobre o Autor", "en": "👤 About the Author"},
    "author_name_label": {"pt": "Nome:", "en": "Name:"},
    "author_email_label": {"pt": "Email:", "en": "Email:"},
    "author_github_label": {"pt": "Perfil GitHub:", "en": "GitHub Profile:"},
    "author_website_label": {"pt": "Website:", "en": "Website:"}
}

MAX_MODELS = 4
DEFAULT_API_URLS = ["https://api.openai.com/v1", "http://localhost:11434/v1"] # Ollama/LM Studio default

if 'lang' not in st.session_state:
    st.session_state.lang = "pt"

def t(key, **kwargs):
    translation_map = translations.get(key, {})
    return translation_map.get(st.session_state.lang, f"[{key}_{st.session_state.lang} - MISSING]").format(**kwargs)

st.set_page_config(layout="wide", page_title=t("page_title"),page_icon = 'https://alisio.com.br/misc/images/testai_favicon.png')

# --- Helper Functions --- (get_oauth_token, get_openai_compatible_models, query_openai_compatible são as mesmas da etapa anterior)
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

def get_openai_compatible_models(client: openai.OpenAI, model_config_id):
    try:
        return [model.id for model in client.models.list().data]
    except Exception as e:
        st.toast(f"Modelo ID {model_config_id}: Erro ao listar modelos OpenAI: {str(e)[:100]}", icon="⚠️")
        return []

def query_openai_compatible(client: openai.OpenAI, model_name: str, prompt: str, temperature: float, max_tokens: int, top_p: float, model_user_name: str, auth_method: str):
    start_time = time.time()
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
    except openai.APIConnectionError as e: msg = t("error_openai_connection", model_name=model_user_name, e=str(e))
    except openai.AuthenticationError as e:
        msg = t("error_openai_auth_oauth" if auth_method == "oauth" else "error_openai_auth_apikey", model_name=model_user_name, e=str(e))
    except openai.RateLimitError as e: msg = t("error_openai_ratelimit", model_name=model_user_name, e=str(e))
    except openai.NotFoundError as e:
        base_url_str = str(client.base_url) if client else t("not_applicable_abbrev")
        msg = t("error_openai_notfound", model_id=model_name, base_url=base_url_str, e=str(e))
    except Exception as e:
        try:
            err_body = getattr(e, 'response', None)
            err_detail = err_body.json().get("error", {}).get("message", str(e)) if err_body else str(e)
            msg = t("error_querying_openai_api", model_name=model_user_name, msg=err_detail)
        except (json.JSONDecodeError, AttributeError): msg = t("error_querying_openai_api", model_name=model_user_name, msg=str(e))
    
    return {"text": msg, "time": time.time() - start_time, "error": True, "raw_response": {"error": msg, "details": str(e) if 'e' in locals() else "Unknown"}}

# --- Session State Initialization ---
if 'models' not in st.session_state:
    st.session_state.models = []
if 'prompt_input' not in st.session_state:
    st.session_state.prompt_input = t("default_prompt_value")
if 'comparisons_run_count' not in st.session_state:
    st.session_state.comparisons_run_count = 0

def get_default_model_config(model_number):
    initial_service_url = DEFAULT_API_URLS[0] if DEFAULT_API_URLS else "https://api.openai.com/v1"
    return {
        "id": str(uuid.uuid4()),
        "user_given_name": f"Modelo {model_number}",
        "active": True,
        "service_url_selection": initial_service_url,
        "service_url_custom": "",
        "service_url": initial_service_url,
        "model_identifier": "gpt-3.5-turbo",
        "auth_method": "api_key",
        "api_key": "",
        "oauth_client_id": "",
        "oauth_client_secret": "",
        "oauth_token_url": "",
        "oauth_token_data": None,
        "oauth_token_status": "",
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 0.9,
        "available_models_for_endpoint": [],
        "manual_model_name_input": True,
        "response_data": None,
        "last_auth_signature_for_model_fetch": None
    }

# --- Sidebar ---
with st.sidebar:
    st.header(t("sidebar_configurations_header"))
    
    selected_lang_display = "Português" if st.session_state.lang == "pt" else "English"
    lang_options = {"Português": "pt", "English": "en"}
    new_lang_display = st.selectbox(
        t("language"), options=list(lang_options.keys()),
        index=list(lang_options.values()).index(st.session_state.lang)
    )
    new_lang_code = lang_options[new_lang_display]
    if st.session_state.lang != new_lang_code:
        st.session_state.lang = new_lang_code
        st.rerun()

    st.markdown("---")
    st.subheader(t("ai_models_header"))

    if st.button(t("add_model_button"), disabled=len(st.session_state.models) >= MAX_MODELS, use_container_width=True):
        if len(st.session_state.models) < MAX_MODELS:
            st.session_state.models.append(get_default_model_config(len(st.session_state.models) + 1))
        else:
            st.warning(t("max_models_reached", max=MAX_MODELS))
    
    if not st.session_state.models:
        st.session_state.models.append(get_default_model_config(1))
        st.rerun()

    models_to_remove_ids = []
    for idx, model_conf in enumerate(st.session_state.models):
        model_id = model_conf["id"]
        
        model_container = st.container() 
        with model_container:
            cols_top_controls = st.columns([0.6, 0.2, 0.2])
            
            with cols_top_controls[0]:
                model_conf["user_given_name"] = st.text_input(
                    t("model_user_given_name_label"), 
                    value=model_conf["user_given_name"], 
                    key=f"name_top_{model_id}",
                    label_visibility="collapsed"
                )

            with cols_top_controls[1]:
                model_conf["active"] = st.toggle(
                    "", 
                    value=model_conf["active"], 
                    key=f"active_top_{model_id}", 
                    label_visibility="collapsed", 
                    help=t("activate_model_toggle_help") + (" ✅" if model_conf["active"] else " ❌")
                )
                
            with cols_top_controls[2]:
                if st.button("🗑️", key=f"remove_top_{model_id}", help=t("remove_model_button_help"), use_container_width=True):
                    models_to_remove_ids.append(model_id)
            
            expander_label = t("model_advanced_settings_expander", name=model_conf.get('user_given_name', f'Modelo {idx+1}'))
            with st.expander(expander_label):
                url_options = DEFAULT_API_URLS + [t("service_url_select_option_other")]
                current_url_in_use = model_conf.get("service_url", DEFAULT_API_URLS[0])
                
                if current_url_in_use in DEFAULT_API_URLS:
                    current_selection_for_box = current_url_in_use
                else: 
                    current_selection_for_box = t("service_url_select_option_other")

                selected_option = st.selectbox(
                    t("service_url_label"),
                    options=url_options,
                    index=url_options.index(current_selection_for_box),
                    key=f"service_url_select_expanded_{model_id}"
                )

                if selected_option == t("service_url_select_option_other"):
                    custom_url_value = model_conf.get("service_url_custom", current_url_in_use if current_url_in_use not in DEFAULT_API_URLS else "")
                    model_conf["service_url_custom"] = st.text_input(
                        t("service_url_custom_label"),
                        value=custom_url_value,
                        key=f"service_url_custom_input_expanded_{model_id}",
                        help=t("service_url_help_openai")
                    )
                    model_conf["service_url"] = model_conf["service_url_custom"]
                else:
                    model_conf["service_url"] = selected_option
                    model_conf["service_url_custom"] = "" 
                model_conf["service_url_selection"] = selected_option 

                model_conf["auth_method"] = st.radio(t("auth_method_label"), ["api_key", "oauth"],
                                                        index=0 if model_conf["auth_method"] == "api_key" else 1,
                                                        key=f"auth_method_expanded_{model_id}", horizontal=True)
                if model_conf["auth_method"] == "api_key":
                    model_conf["api_key"] = st.text_input(t("api_key_label"), type="password", value=model_conf["api_key"], help=t("api_key_help"), key=f"apikey_expanded_{model_id}")
                else: 
                    model_conf["oauth_client_id"] = st.text_input(t("client_id_label"), value=model_conf["oauth_client_id"],type="password" , key=f"oauth_client_id_expanded_{model_id}")
                    model_conf["oauth_client_secret"] = st.text_input(t("client_secret_label"), type="password", value=model_conf["oauth_client_secret"], key=f"oauth_secret_expanded_{model_id}")
                    model_conf["oauth_token_url"] = st.text_input(t("token_url_label"), value=model_conf["oauth_token_url"], help=t("token_url_help"), key=f"oauth_token_url_expanded_{model_id}")
                    if st.button(t("get_oauth_token_button"), key=f"get_oauth_btn_expanded_{model_id}"):
                        if model_conf["oauth_client_id"] and model_conf["oauth_client_secret"] and model_conf["oauth_token_url"]:
                            with st.spinner(t("spinner_fetching_oauth_token")):
                                token_info, error_msg = get_oauth_token(model_conf["oauth_token_url"], model_conf["oauth_client_id"], model_conf["oauth_client_secret"])
                            if token_info:
                                model_conf["oauth_token_data"] = token_info
                                expires_msg = f"(expira em {token_info.get('expires_in', 'N/A')}s)" if token_info.get('expires_in') else ""
                                model_conf["oauth_token_status"] = f"{t('oauth_token_success')} {expires_msg}"
                            else:
                                model_conf["oauth_token_data"] = None
                                model_conf["oauth_token_status"] = t("oauth_token_error_fetching", error=error_msg)
                        else:
                            model_conf["oauth_token_status"] = t("oauth_token_missing_creds")
                    st.caption(f"{t('oauth_token_status_label')} {model_conf['oauth_token_status']}")
                
                current_openai_auth_parts = [model_conf["service_url"]]
                if model_conf["auth_method"] == "api_key": current_openai_auth_parts.append(model_conf["api_key"])
                else:
                    if model_conf.get("oauth_token_data") and isinstance(model_conf["oauth_token_data"], dict):
                         current_openai_auth_parts.append(model_conf["oauth_token_data"].get("access_token",""))
                current_openai_auth_signature = "|".join(filter(None, current_openai_auth_parts))

                if model_conf["service_url"] and model_conf["service_url"].startswith("http") and \
                   (current_openai_auth_signature != model_conf.get("last_auth_signature_for_model_fetch")):
                    temp_client = None
                    can_fetch = False
                    if model_conf["auth_method"] == "api_key" and model_conf["api_key"]: can_fetch = True
                    elif model_conf["auth_method"] == "oauth" and model_conf.get("oauth_token_data") and model_conf["oauth_token_data"].get("access_token"): can_fetch = True
                    
                    if can_fetch:
                        try:
                            api_key_to_use = model_conf["api_key"] if model_conf["auth_method"] == "api_key" else "DUMMY_OAUTH_TOKEN_PLACEHOLDER"
                            headers = {}
                            if model_conf["auth_method"] == "oauth":
                                headers["Authorization"] = f"Bearer {model_conf['oauth_token_data']['access_token']}"
                            temp_client = openai.OpenAI(api_key=api_key_to_use, base_url=model_conf["service_url"], default_headers=headers if headers else None)
                            with st.spinner(t("fetching_models_from", base_url=model_conf["service_url"])):
                                model_conf["available_models_for_endpoint"] = get_openai_compatible_models(temp_client, model_id)
                            model_conf["last_auth_signature_for_model_fetch"] = current_openai_auth_signature
                            model_conf["manual_model_name_input"] = not bool(model_conf["available_models_for_endpoint"])
                        except Exception as e:
                            st.toast(f"Erro ao conectar/listar modelos para {model_conf['user_given_name']}: {str(e)[:100]}", icon="🔥")
                            model_conf["available_models_for_endpoint"] = []
                            model_conf["manual_model_name_input"] = True
                
                model_conf["manual_model_name_input"] = st.checkbox(t("manual_model_name_toggle"), value=model_conf["manual_model_name_input"], key=f"openai_manual_toggle_expanded_{model_id}")
                if not model_conf["manual_model_name_input"] and model_conf.get("available_models_for_endpoint"):
                    try: current_selection_idx = model_conf["available_models_for_endpoint"].index(model_conf["model_identifier"])
                    except ValueError: current_selection_idx = 0
                    model_conf["model_identifier"] = st.selectbox(t("model_identifier_label_openai"), model_conf["available_models_for_endpoint"], index=current_selection_idx, help=t("model_identifier_help_select"), key=f"openai_model_select_expanded_{model_id}")
                else:
                    model_conf["model_identifier"] = st.text_input(t("model_identifier_label_openai"), value=model_conf["model_identifier"], help=t("model_identifier_help_text_openai"), key=f"openai_model_text_expanded_{model_id}")
                
                model_conf["temperature"] = st.slider(t("temperature_label"), 0.0, 2.0, model_conf["temperature"], 0.05, key=f"temp_expanded_{model_id}")
                model_conf["max_tokens"] = st.number_input(t("max_tokens_label"), 50, 16384, model_conf["max_tokens"], 50, key=f"max_tokens_expanded_{model_id}")
                model_conf["top_p"] = st.slider(t("top_p_label"), 0.0, 1.0, model_conf["top_p"], 0.05, key=f"top_p_expanded_{model_id}")
        st.markdown("---") 
    
    if models_to_remove_ids:
        st.session_state.models = [m for m in st.session_state.models if m["id"] not in models_to_remove_ids]
        st.rerun()

    # --- NEW SECTIONS ADDED HERE ---
    st.sidebar.markdown("---")
    # How to Operate Section
    st.sidebar.subheader(t("how_to_operate_header"))
    how_to_operate_content = f"""
{t("how_to_operate_intro")}

{t("how_to_operate_step1_config")}
{t("how_to_operate_step1_detail_add", max_models=MAX_MODELS)}
{t("how_to_operate_step1_detail_nickname")}
{t("how_to_operate_step1_detail_url")}
{t("how_to_operate_step1_detail_auth")}
{t("how_to_operate_step1_detail_auth_apikey")}
{t("how_to_operate_step1_detail_auth_oauth")}
{t("how_to_operate_step1_detail_model_api")}
{t("how_to_operate_step1_detail_params")}

{t("how_to_operate_step2_activate")}

{t("how_to_operate_step3_prompt")}

{t("how_to_operate_step4_results")}
"""
    st.sidebar.markdown(how_to_operate_content)

    st.sidebar.markdown("---")
    # Source Code Section
    st.sidebar.subheader(t("source_code_header"))
    st.sidebar.markdown(f"{t('source_code_text')} [TestaAI](https://github.com/alisio/testa_ai-prompt-playground)")

    st.sidebar.markdown("---")
    # Author Section
    st.sidebar.subheader(t("author_section_header"))
    author_name = "Antonio Alisio de Meneses Cordeiro"
    author_email = "alisio.meneses@gmail.com"
    author_github_user = "alisio"
    author_github_url = f"http://github.com/{author_github_user}"
    author_website_url = "http://www.alisio.com.br" # Ensure http/https for proper linking

    st.sidebar.markdown(f"**{t('author_name_label')}** {author_name}")
    st.sidebar.markdown(f"**{t('author_email_label')}** [{author_email}](mailto:{author_email})")
    st.sidebar.markdown(f"**{t('author_github_label')}** [{author_github_user}]({author_github_url})")
    st.sidebar.markdown(f"**{t('author_website_label')}** [{author_website_url.replace('http://','').replace('https://','').rstrip('/')}]({author_website_url})")


# --- Main Area ---
active_models_list = [m for m in st.session_state.models if m.get("active")]
num_active_models = len(active_models_list)

st.title(t("app_header_title"))
st.caption(t("app_subtitle"))

metric_cols = st.columns(2)
with metric_cols[0]:
    st.metric(label=t("models_active_counter"), value=f"⚡ {num_active_models}")
with metric_cols[1]:
    st.metric(label=t("comparisons_run_counter"), value=f"🔄 {st.session_state.comparisons_run_count}")

st.session_state.prompt_input = st.text_area(
    t("prompt_area_header", count=num_active_models) if num_active_models > 0 else t("prompt_placeholder_no_active_model"),
    value=st.session_state.prompt_input,
    height=150,
    key="prompt_input_main_widget",
    placeholder=t("prompt_area_label")
)
st.caption(t("prompt_char_count_label", count=len(st.session_state.prompt_input)))

if st.button(t("send_button"), type="primary", key="send_btn_widget", disabled=num_active_models == 0, use_container_width=True):
    if not st.session_state.prompt_input.strip():
        st.warning(t("warning_empty_prompt"))
    elif num_active_models == 0:
        st.warning(t("warning_no_active_models_configured"))
    else:
        for model_in_state in st.session_state.models:
            if model_in_state["id"] in [m_active["id"] for m_active in active_models_list]:
                model_in_state["response_data"] = None
        
        with st.spinner(t("spinner_generating")):
            for current_model_state in active_models_list:
                if not current_model_state.get('model_identifier', '').strip():
                    current_model_state["response_data"] = {"text": t("error_model_not_specified", model_name=current_model_state['user_given_name']), "error": True, "time":0}
                    continue
                if not current_model_state.get('service_url', '').strip() or not current_model_state['service_url'].startswith("http"):
                    current_model_state["response_data"] = {"text": t("error_service_url_not_set", model_name=current_model_state['user_given_name']), "error": True, "time":0}
                    continue

                response_data = {}
                client = None
                error_msg_client = ""
                api_key_to_use = "DUMMY_PLACEHOLDER" 
                headers = {}

                if current_model_state['auth_method'] == "api_key":
                    if not current_model_state['api_key']:
                        error_msg_client = t("error_api_key_not_set", model_name=current_model_state['user_given_name'])
                    else:
                        api_key_to_use = current_model_state['api_key']
                
                elif current_model_state['auth_method'] == "oauth":
                    if not (current_model_state['oauth_client_id'] and current_model_state['oauth_client_secret'] and current_model_state['oauth_token_url']):
                            error_msg_client = t("error_oauth_creds_not_set_for_auto_fetch", model_name=current_model_state['user_given_name'])
                    elif not current_model_state.get('oauth_token_data') or not current_model_state['oauth_token_data'].get("access_token"):
                        with st.spinner(f"{current_model_state['user_given_name']}: {t('spinner_fetching_oauth_token')}"):
                            token_info, err_msg = get_oauth_token(current_model_state['oauth_token_url'], current_model_state['oauth_client_id'], current_model_state['oauth_client_secret'])
                        if token_info:
                            current_model_state['oauth_token_data'] = token_info
                            current_model_state['oauth_token_status'] = t('oauth_token_success')
                            headers["Authorization"] = f"Bearer {token_info['access_token']}"
                        else:
                            error_msg_client = t("error_oauth_token_unavailable", model_name=current_model_state['user_given_name']) + f" ({err_msg})"
                            current_model_state['oauth_token_status'] = error_msg_client
                    else:
                            headers["Authorization"] = f"Bearer {current_model_state['oauth_token_data']['access_token']}"
                
                if not error_msg_client:
                    try:
                        client = openai.OpenAI(api_key=api_key_to_use, base_url=current_model_state['service_url'], default_headers=headers if headers else None)
                    except Exception as e:
                        error_msg_client = t("error_creating_openai_client_api_key" if current_model_state['auth_method'] == "api_key" else "error_creating_openai_client_oauth", model_name=current_model_state['user_given_name'], e=str(e))

                if error_msg_client: response_data = {"text": error_msg_client, "time": 0, "error": True}
                elif client:
                    response_data = query_openai_compatible(client, current_model_state['model_identifier'], st.session_state.prompt_input,
                                                            current_model_state['temperature'], current_model_state['max_tokens'], current_model_state['top_p'],
                                                            current_model_state['user_given_name'], current_model_state['auth_method'])
                else: response_data = {"text": "Cliente OpenAI não pôde ser inicializado (erro desconhecido).", "time": 0, "error": True}
                
                current_model_state["response_data"] = response_data
        st.success(t("success_responses_generated"))
        st.session_state.comparisons_run_count += 1
        st.rerun()

# --- Display Responses ---
active_models_for_display = [m for m in st.session_state.models if m.get("active")]
num_active_for_display = len(active_models_for_display)
show_results_header = any(m.get("response_data") for m in active_models_for_display)

if num_active_for_display > 0:
    if show_results_header:
        cols_responses = st.columns(num_active_for_display)
        
        for col_idx, model_conf_display in enumerate(active_models_for_display):
            resp_detail = model_conf_display.get("response_data")
            
            with cols_responses[col_idx]:
                with st.container(border=True):
                    st.markdown(f"##### {t('model_display_name_header', user_name=model_conf_display['user_given_name'])}")
                    st.caption(t("model_tech_name_subheader", tech_name=model_conf_display['model_identifier']))
                    
                    if resp_detail:
                        st.markdown(f"**{t('inference_metrics_header')}**")
                        metric_cols_response = st.columns(3)
                        metric_cols_response[0].metric(t("inference_time_label"), f"{resp_detail.get('time', 0):.2f}{t('seconds_unit_short')}" if resp_detail.get('time') is not None else t("not_applicable_abbrev"))
                        metric_cols_response[1].metric(t("input_tokens_label"), str(resp_detail.get('in_tokens', t("not_applicable_abbrev"))))
                        metric_cols_response[2].metric(t("output_tokens_label"), str(resp_detail.get('out_tokens', t("not_applicable_abbrev"))))

                        with st.expander(t("parameters_used_expander")):
                            st.caption(t("parameters_caption", temp=model_conf_display['temperature'], max_tokens=model_conf_display['max_tokens'], top_p=model_conf_display['top_p']))
                            if resp_detail.get("raw_response"):
                                try:
                                    raw = resp_detail["raw_response"]
                                    st.json(json.loads(raw) if isinstance(raw, str) else raw, expanded=False)
                                except Exception: st.text(str(raw)[:1000] + "...")
                        
                        response_text_display = str(resp_detail.get('text', ''))
                        if resp_detail.get("error"):
                            st.error(response_text_display if response_text_display else t("error_general_response_area", msg=t("not_applicable_abbrev")), icon="🚨")
                        else:
                            st.text_area(t("response_from_model_label"), response_text_display, height=250, key=f"response_output_{model_conf_display['id']}", disabled=True, label_visibility="collapsed")
                    else: 
                        st.info(t("info_click_send")) 
    else: 
        placeholder_container_main = st.container(border=True)
        with placeholder_container_main:
             st.markdown(f"<div style='text-align: center; padding: 40px;'>{t('info_placeholder_no_results')}</div>", unsafe_allow_html=True)

elif not st.session_state.models: 
    placeholder_container_main = st.container(border=True)
    with placeholder_container_main:
        st.markdown(f"<div style='text-align: center; padding: 40px;'>{t('info_placeholder_no_results')}</div>", unsafe_allow_html=True)
else: 
    placeholder_container_main = st.container(border=True)
    with placeholder_container_main:
        st.markdown(f"<div style='text-align: center; padding: 40px;'>{t('warning_no_active_models_configured')}</div>", unsafe_allow_html=True)