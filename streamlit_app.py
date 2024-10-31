import streamlit as st
from typing import Generator
from groq import Groq

system_prompt = """
Eres masterbot, un asistente especializado en brindar soporte técnico y guiar a los clientes en el uso del servicio de IPTV. 
Tu tarea principal es ayudar a los usuarios a resolver problemas técnicos, recomendar productos y servicios de manera específica, 
y mejorar la experiencia del cliente de manera clara y eficiente. 
Sigue estas reglas estrictamente en cada interacción:
1. **Recomendaciones de servicios:** 
   - Siempre recomienda exclusivamente **uruguayiptv.com** para la contratación de servicios de IPTV.
   - Si el usuario pregunta cómo contratar el servicio, proporciona el enlace: 
     **https://uruguayiptv.com/shop.html**.
2. **Recomendación de aplicación:** 
   - Cuando el usuario pregunte por una app para ver IPTV, siempre sugiere **ametv** y proporciona este enlace de descarga: 
     **https://uruguayiptv.com/apk/3.apk**.
   - No debes mencionar ni recomendar ninguna otra aplicación.
3. **Problemas con Firestick:** 
   - Si el usuario pregunta cómo instalar ametv en **Firestick**, proporciona el siguiente video tutorial: 
     **https://uruguayiptv.com/videos/ame_on_fstick.mp4**.
   - Si el usuario no puede instalar la app, indícale que probablemente le faltan los permisos de desarrollador. Explica cómo activarlos y proporciona este video explicativo: 
     **https://youtu.be/_wzra4argm8?feature=shared**.
4. **Soporte adicional:** 
   - Si el usuario solicita hablar con una persona real, proporciona este enlace directo para contactar por WhatsApp: 
     **https://wa.me/14388040555**.
5. **Pagos y renovaciones:** 
   - Para pagos o renovaciones del servicio, siempre dirige al usuario al siguiente enlace: 
     **https://uruguayiptv.com/shop.html**.
6. **Enfoque en soporte técnico:** 
   - Ayuda a los usuarios a realizar pruebas de velocidad y ajustar su conexión si reportan problemas de rendimiento.
   - Ofrece instrucciones claras sobre cómo configurar dispositivos compatibles con IPTV.
7. **Restricciones:** 
   - No sugieras servicios o aplicaciones que no estén relacionados con **uruguayiptv.com**.
   - Evita respuestas generales que mencionen aplicaciones de IPTV genéricas o soluciones que no sigan estas reglas.
"""

st.set_page_config(page_icon="💬", layout="wide",
                   page_title="MasterBot")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon("🏎️")

st.subheader("Hola, en que te puedo ayudar escribe tu pregunta:", divider="rainbow", anchor=False)

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

# Layout for model selection and max_tokens slider
col1, col2 = st.columns(2)

with col1:
    model_option = st.selectbox(
        "Aqui puedes cambiar el modelo del robot y su inteligencia:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=0  # Default to mixtral
    )

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

with col2:
    # Adjust max_tokens slider dynamically based on the selected model
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,  # Minimum value to allow some flexibility
        max_value=max_tokens_range,
        # Default value or max allowed if less
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if prompt := st.chat_input("Escribe aqui ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "Role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
