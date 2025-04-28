import openai
import pandas as pd
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os


load_dotenv()

# Configuração da API OpenAI
OPENAI_API_KEY = os.getenv("API_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)


# Função de classificação
def classificar_texto(texto, max_retries=3, wait_time=5):
    if texto is None or str(texto).strip() in ["", ".", "*"]:
        return "Não classificado"

    prompt = f"""
        Analise as justificativas de hora extra abaixo.
        
        Para cada justificativa:
        - Classifique em UMA das categorias abaixo:
        
        Categorias:
        1. Parada de Máquina
        2. Manutenção/Inspeção/Teste de Máquina
        3. Substituição de Colaborador
        4. Outros Eventos Externos
        5. Não Classificado
        
        Instruções:
        - Cite apenas a categoria (ex: "Parada de Máquina").
        - NÃO explique o motivo.
        - Se o texto mencionar vários aspectos, classifique pelo PRINCIPAL.
        
        Exemplos:
        11. ACOMPANHAMENTO DO FORNO 2 --> Manutenção/Inspeção/Teste de Máquina
        12. PARADA DO FORNO 2 --> Parada de Máquina
        13. SUBSTITUIÇÃO DE AUSÊNCIA --> Substituição de Colaborador
        14. ELEIÇÕES MUNICIPAIS 2024 --> Outros Eventos Externos
        15. ORGANIZAÇÃO DO GALPÃO --> Outros Eventos Externos
        
        Agora classifique:
        
        {texto}
        """

    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "text"},
                temperature=0.3,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            resposta_texto = response.choices[0].message.content.strip()
            return resposta_texto
        except openai.OpenAIError as e:
            print(
                f"Tentativa {attempt + 1} falhou: {str(e)}. Retentando em {wait_time} segundos..."
            )
            attempt += 1
            time.sleep(wait_time)

    return f"Erro após {max_retries} tentativas"


# ----------- Início do Script ---------------

# Lê o CSV
df = pd.read_csv("justificativas_200.csv")

# Cria colunas para armazenar classificações
motivos_classificados = []

# Iterar linha a linha (com barra de progresso)
for _, row in tqdm(df.iterrows(), total=len(df)):
    motivo = classificar_texto(row["Justificativa_processada"])
    motivos_classificados.append(motivo)
    time.sleep(2)  # Sleep para evitar sobrecarga na API

# Adiciona resultados no DataFrame
df["Justificativa_classificada"] = motivos_classificados

# Salva o resultado em CSV
df.to_csv("justificativas_classificadas.csv", index=False)

print("Classificação finalizada e salva em justificativas_classificadas.csv ✅")
