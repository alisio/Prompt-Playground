import streamlit as st
import openai
import ollama
import json # Para tratar possíveis erros de JSON em respostas

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="LLM Prompt Comparator")

# --- Funções Auxiliares para Chamadas de API ---

def get_ollama_models(base_url):
    """Lista modelos disponíveis em um endpoint Ollama."""
    try:
        client = ollama.Client(host=base_url)
        models_info = client.list()
        return [model['model'] for model in models_info['models']]
    except Exception as e:
        st.sidebar.error(f"Ollama ({base_url}): Erro ao listar modelos: {e}")
        return []

def query_ollama(base_url, model_name, prompt, temperature, max_tokens, top_p):
    """Envia um prompt para um modelo Ollama."""
    try:
        client = ollama.Client(host=base_url)
        response = client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': temperature,
                'num_predict': max_tokens, # Em Ollama, num_predict é análogo a max_tokens
                'top_p': top_p,
            }
        )
        return response['message']['content']
    except Exception as e:
        return f"Erro ao consultar Ollama ({model_name}): {e}"

def get_openai_compatible_models(api_key, base_url):
    """
    Tenta listar modelos de um endpoint OpenAI compatível.
    Muitos endpoints privados podem não suportar isso ou requerer permissões específicas.
    """
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception:
        # Se falhar (comum para endpoints privados não-OpenAI), retorna lista vazia
        # O usuário precisará digitar o nome do modelo manualmente.
        return []


def query_openai_compatible(api_key, base_url, model_name, prompt, temperature, max_tokens, top_p):
    """Envia um prompt para um endpoint compatível com OpenAI."""
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        return completion.choices[0].message.content
    except openai.APIConnectionError as e:
        return f"Erro de conexão com API OpenAI ({model_name}): {e}"
    except openai.AuthenticationError as e:
        return f"Erro de autenticação com API OpenAI ({model_name}): Chave inválida ou não fornecida? {e}"
    except openai.RateLimitError as e:
        return f"Erro de limite de taxa com API OpenAI ({model_name}): {e}"
    except openai.NotFoundError as e:
         return f"Erro: Modelo '{model_name}' não encontrado no endpoint '{base_url}'. Verifique o nome. {e}"
    except Exception as e:
        try:
            # Tentar extrair mensagem de erro mais detalhada se for um erro da API
            error_details = json.loads(str(e)) # Muitas APIs retornam JSON no corpo do erro
            msg = error_details.get("error", {}).get("message", str(e))
            return f"Erro ao consultar API OpenAI ({model_name}): {msg}"
        except:
            return f"Erro ao consultar API OpenAI ({model_name}): {e}"


# --- Interface Streamlit ---
st.title("🧪 LLM Prompt Comparator Dashboard")
st.markdown("Desenvolva e compare prompts em diferentes modelos e endpoints LLM.")

# --- Estado da Sessão para Armazenar Respostas ---
if 'responses' not in st.session_state:
    st.session_state.responses = [None, None, None]
if 'model_configs' not in st.session_state:
    st.session_state.model_configs = [{}, {}, {}] # Para armazenar configurações de cada modelo

# --- Sidebar para Configurações Globais e por Modelo ---
with st.sidebar:
    st.header("⚙️ Configurações Gerais")

    # Credenciais e URLs Base
    st.subheader("🔑 Credenciais e Endpoints")
    ollama_base_url = st.text_input("Ollama Base URL", value="http://localhost:11434", key="ollama_url")
    
    openai_api_key = st.text_input("OpenAI-compatível API Key", type="password", key="openai_key", help="Pode ser 'NA' ou qualquer string se o endpoint não exigir chave.")
    openai_base_url = st.text_input("OpenAI-compatível Base URL", key="openai_base_url", help="Ex: https://api.openai.com/v1 ou URL do seu endpoint privado")

    # Parâmetros de Inferência Globais (podem ser sobrescritos por modelo se desejado)
    st.subheader("🧠 Parâmetros de Inferência")
    global_temperature = st.slider("Temperatura", 0.0, 1.0, 0.7, 0.05, key="global_temp")
    global_max_tokens = st.number_input("Max Tokens", 50, 4096, 512, 50, key="global_max_tokens")
    global_top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05, key="global_top_p")

    st.markdown("---")
    
    num_models_to_compare = st.radio(
        "Comparar Modelos:",
        (1, 2, 3),
        index=1, # Default para 2 modelos
        horizontal=True,
        key="num_models"
    )
    st.markdown("---")

    # Configuração por Modelo
    model_configs_ui = []
    available_ollama_models = get_ollama_models(ollama_base_url) if ollama_base_url else []
    
    # Cache para modelos OpenAI para não chamar API toda hora se URL não mudar
    if 'last_openai_base_url' not in st.session_state:
        st.session_state.last_openai_base_url = ""
    if 'available_openai_models' not in st.session_state:
        st.session_state.available_openai_models = []

    if openai_base_url and openai_api_key and openai_base_url != st.session_state.get('last_openai_base_url', ""):
        with st.spinner(f"Buscando modelos de {openai_base_url}..."):
            st.session_state.available_openai_models = get_openai_compatible_models(openai_api_key, openai_base_url)
            st.session_state.last_openai_base_url = openai_base_url
    
    available_openai_models_cached = st.session_state.available_openai_models


    for i in range(num_models_to_compare):
        st.header(f"Modelo {i+1}")
        config = {}
        config['active'] = st.checkbox(f"Ativar Modelo {i+1}", value=True, key=f"active_{i}")
        
        if config['active']:
            config['endpoint_type'] = st.selectbox(
                f"Tipo de Endpoint (Modelo {i+1})",
                ("Ollama", "OpenAI-compatível"),
                key=f"endpoint_type_{i}"
            )

            if config['endpoint_type'] == "Ollama":
                if available_ollama_models:
                    config['model_name'] = st.selectbox(
                        f"Modelo Ollama (Modelo {i+1})",
                        options=available_ollama_models,
                        index=0 if available_ollama_models else None,
                        key=f"ollama_model_{i}"
                    )
                else:
                    config['model_name'] = st.text_input(
                        f"Nome do Modelo Ollama (Modelo {i+1})",
                        help="Ex: llama3:latest. Modelos não puderam ser listados.",
                        key=f"ollama_model_text_{i}"
                    )
            else: # OpenAI-compatível
                if available_openai_models_cached:
                    config['model_name'] = st.selectbox(
                        f"Modelo OpenAI (Modelo {i+1})",
                        options=available_openai_models_cached,
                        index=0 if available_openai_models_cached else None,
                        help="Selecione ou digite o nome do modelo se não estiver na lista.",
                        key=f"openai_model_{i}"
                    )
                    # Adicionar opção para digitar manualmente se necessário, ou se a lista estiver vazia
                    if st.checkbox(f"Digitar nome do modelo OpenAI manualmente (Modelo {i+1})", key=f"openai_manual_toggle_{i}") or not available_openai_models_cached :
                         config['model_name'] = st.text_input(
                            f"Nome do Modelo OpenAI (Modelo {i+1})",
                            value=config.get('model_name', "gpt-3.5-turbo"),
                            help="Ex: gpt-4, gpt-3.5-turbo, ou nome do seu modelo privado",
                            key=f"openai_model_text_{i}"
                        )
                else:
                     config['model_name'] = st.text_input(
                        f"Nome do Modelo OpenAI (Modelo {i+1})",
                        value=config.get('model_name', "gpt-3.5-turbo"),
                        help="Ex: gpt-4, gpt-3.5-turbo, ou nome do seu modelo privado. Modelos não puderam ser listados.",
                        key=f"openai_model_text_alt_{i}"
                    )


            # Parâmetros específicos (opcional, sobrescreve globais)
            # Por simplicidade, vamos usar os globais, mas aqui seria o local para sliders por modelo
            config['temperature'] = global_temperature
            config['max_tokens'] = global_max_tokens
            config['top_p'] = global_top_p
        
        model_configs_ui.append(config)
        st.markdown("---")
    
    st.session_state.model_configs = model_configs_ui


# --- Área Principal para Prompt e Respostas ---
st.subheader("💬 Prompt")
prompt_text = st.text_area("Digite seu prompt aqui:", height=150, key="prompt_input", value="Write a poem about Python")

if st.button("🚀 Gerar Respostas", type="primary"):
    if not prompt_text.strip():
        st.warning("Por favor, insira um prompt.")
    else:
        st.session_state.responses = [None] * 3 # Limpa respostas anteriores
        with st.spinner("Gerando respostas..."):
            active_configs = [conf for conf in st.session_state.model_configs if conf.get('active', False)]
            
            for idx, config in enumerate(active_configs):
                if not config.get('model_name', '').strip():
                    st.session_state.responses[idx] = f"Modelo {idx+1}: Nome do modelo não especificado."
                    continue

                response_text = ""
                if config['endpoint_type'] == "Ollama":
                    if not ollama_base_url:
                        st.session_state.responses[idx] = f"Modelo {idx+1} (Ollama): URL base do Ollama não configurada."
                        continue
                    response_text = query_ollama(
                        ollama_base_url,
                        config['model_name'],
                        prompt_text,
                        config['temperature'],
                        config['max_tokens'],
                        config['top_p']
                    )
                elif config['endpoint_type'] == "OpenAI-compatível":
                    if not openai_api_key or not openai_base_url:
                        st.session_state.responses[idx] = f"Modelo {idx+1} (OpenAI): API Key ou Base URL não configurados."
                        continue
                    response_text = query_openai_compatible(
                        openai_api_key,
                        openai_base_url,
                        config['model_name'],
                        prompt_text,
                        config['temperature'],
                        config['max_tokens'],
                        config['top_p']
                    )
                st.session_state.responses[idx] = response_text
        st.success("Respostas geradas!")

# --- Exibição das Respostas ---
st.subheader("📊 Resultados da Comparação")

active_model_indices = [i for i, conf in enumerate(st.session_state.model_configs) if conf.get('active', False) and i < num_models_to_compare]

if any(st.session_state.responses[i] for i in active_model_indices):
    # Criar colunas dinamicamente com base no número de modelos ativos selecionados para comparação
    cols = st.columns(len(active_model_indices) if active_model_indices else 1)
    
    col_idx = 0
    for i in active_model_indices:
        config = st.session_state.model_configs[i]
        response = st.session_state.responses[i]
        
        if response: # Apenas mostra se há uma resposta (ou erro) para este slot
            with cols[col_idx]:
                model_display_name = config.get('model_name', f"Modelo {i+1} (Não Configurado)")
                if not config.get('model_name'): # Se o nome do modelo estiver vazio mas ativo
                    model_display_name = f"Modelo {i+1} ({config.get('endpoint_type', 'N/A')})"

                st.markdown(f"#### Modelo {i+1}: {model_display_name}")
                st.markdown(f"**Endpoint:** `{config.get('endpoint_type', 'N/A')}`")
                
                # Mostrar parâmetros usados
                with st.expander("Parâmetros Usados"):
                    st.caption(f"Temp: {config.get('temperature', global_temperature)}, Max Tokens: {config.get('max_tokens', global_max_tokens)}, Top P: {config.get('top_p', global_top_p)}")

                # Usar st.text_area com disabled=True para permitir seleção e cópia fácil
                st.text_area(
                    f"Resposta de {model_display_name}",
                    value=str(response), # Garantir que é string
                    height=400,
                    key=f"response_output_{i}",
                    disabled=True, # Faz parecer um display mas permite seleção
                    label_visibility="collapsed"
                )
            col_idx += 1
    
    if not active_model_indices:
         st.info("Nenhum modelo ativo para exibir resultados. Configure e ative modelos na barra lateral.")

else:
    st.info("Clique em 'Gerar Respostas' após configurar os modelos e inserir um prompt.")

# --- Dicas de Uso ---
st.sidebar.markdown("---")
st.sidebar.info("""
**Como usar:**
1.  Configure as URLs e chaves de API na seção "Credenciais".
2.  Escolha quantos modelos comparar (1 a 3).
3.  Para cada modelo:
    * Ative-o.
    * Selecione o tipo de endpoint.
    * Escolha/digite o nome do modelo.
    * (Opcional) Ajuste parâmetros de inferência (padrão globais).
4.  Digite seu prompt na área principal.
5.  Clique em "Gerar Respostas".
""")