import os
import yaml
from llama_cpp import Llama
from datetime import datetime
import sys
import time

def load_system_prompt():
    """Carrega o system prompt de um arquivo yaml"""
    system_prompt_path = "system_prompt.yaml"
    
    # Verifica se o arquivo existe, se não existe, cria um com um prompt padrão
    if not os.path.exists(system_prompt_path):
        default_prompt = """prompt: |
  You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."""
        with open(system_prompt_path, "w", encoding="utf-8") as f:
            f.write(default_prompt)
        print(f"Arquivo {system_prompt_path} criado com prompt padrão.")
        return "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    
    # Carrega o prompt do arquivo YAML
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f)
            if yaml_content and 'prompt' in yaml_content:
                system_prompt = yaml_content['prompt']
                print(f"System prompt carregado de {system_prompt_path}")
                return system_prompt
            else:
                print(f"Erro: Formato inválido em {system_prompt_path}. Usando prompt padrão.")
                return "You are an AI programming assistant using Deepseek Coder model."
    except Exception as e:
        print(f"Erro ao carregar o system prompt: {e}. Usando prompt padrão.")
        return "You are an AI programming assistant using Deepseek Coder model."

def load_user_input():
    """Carrega o input do usuário de um arquivo yaml"""
    input_path = "input.yaml"
    
    # Verifica se o arquivo existe, se não existe, cria um com um exemplo
    if not os.path.exists(input_path):
        example_input = """input: |
  Escreva uma função em Python que calcule o fatorial de um número."""
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(example_input)
        print(f"Arquivo {input_path} criado com exemplo de pergunta.")
        return "Escreva uma função em Python que calcule o fatorial de um número."
    
    # Carrega o input do arquivo YAML
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f)
            if yaml_content and 'input' in yaml_content:
                user_input = yaml_content['input']
                print(f"Input carregado de {input_path}")
                return user_input
            else:
                print(f"Erro: Formato inválido em {input_path}. Usando input padrão.")
                return "Escreva uma função em Python que calcule o fatorial de um número."
    except Exception as e:
        print(f"Erro ao carregar o input: {e}. Usando input padrão.")
        return "Escreva uma função em Python que calcule o fatorial de um número."

def save_to_log(user_input, response):
    """Salva a interação em um arquivo de log"""
    # Cria a pasta de logs se não existir
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Gera o nome do arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"chat_{timestamp}.md")
    
    # Salva a conversa no arquivo
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"# Conversa {timestamp}\n\n")
        f.write("## Pergunta\n\n")
        f.write(f"{user_input}\n\n")
        f.write("## Resposta\n\n")
        f.write(f"{response}\n")
    
    print(f"\nInteração salva em: {log_file}")

def main():
    # Caminho para o modelo
    model_path = os.path.join("models", "DeepSeek-Coder-V2-Lite-Instruct.IQ3_M.gguf")
    
    # Verifica se o modelo existe
    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em {model_path}")
        return
    
    # Carrega o system prompt
    system_prompt = load_system_prompt()
    
    # Inicializa o modelo com configurações otimizadas para estabilidade
    print("Carregando o modelo (isto pode levar alguns minutos)...")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=8192,  # Reduzido para melhorar a estabilidade
            n_batch=512,  # Tamanho de lote menor para processamento mais estável
            n_gpu_layers=32,  # Limitar camadas na GPU para evitar problemas de memória
            verbose=False,  # Desativa logs verbosos
            seed=42  # Adiciona uma seed para resultados consistentes
        )
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return
    
    # Prompt template para DeepSeek Coder com system prompt carregado do arquivo
    # Usando um formato mais seguro e explícito
    prompt_template = """
{}
### Instrução:
{}
### RESPOSTA:
"""
    
    while True:
        try:
            # Pergunta se o usuário quer processar o arquivo de input
            action = input("\nProcessar pergunta do arquivo input.yaml? (s/n): ")
            
            if action.lower() not in ['s', 'sim', 'y', 'yes']:
                print("Encerrando o programa.")
                break
            
            # Carrega o input do usuário do arquivo
            user_input = load_user_input()
            print("\nPergunta carregada do arquivo input.yaml:")
            print(f"\n{user_input}\n")
            
            # Formata o prompt - evitando o uso de f-strings aninhados que podem causar problemas
            prompt = prompt_template.format(system_prompt, user_input)
            
            # Gera a resposta com streaming token por token
            print("\nGerando resposta...\n")
            
            # Inicializa a resposta completa
            full_response = ""
            
            # Configurações para geração máxima de tokens
            try:
                # Usando streaming para ver os tokens sendo gerados
                completion = llm.create_completion(
                    prompt=prompt,
                    max_tokens=8192,  # Valor máximo seguro (próximo do limite de contexto, deixando espaço para o prompt)
                    temperature=0.5,  # Temperatura reduzida para respostas mais determinísticas
                    top_p=0.9,
                    repeat_penalty=1.1,
                    top_k=40,
                    stop=["### Instrução:", "### RESPOSTA:"],
                    stream=True  # Mantém o streaming ativado conforme solicitado
                )
                
                # Usando streaming para gerar a resposta token por token
                for token in completion:
                    try:
                        # Extrai o texto do token
                        token_text = token['choices'][0]['text']
                        
                        # Adiciona à resposta completa
                        full_response += token_text
                        
                        # Imprime o token imediatamente (sem quebra de linha)
                        print(token_text, end="", flush=True)
                        sys.stdout.flush()  # Garante que o output seja exibido imediatamente
                        
                        # Pequena pausa ampliada para evitar sobrecarga
                        time.sleep(0.002)  # Ligeiramente maior para menos sobrecarga no sistema
                    except Exception as e:
                        print(f"\nErro durante o streaming: {e}")
                        break
                
                print("\n")  # Adiciona uma quebra de linha após a resposta completa
            except Exception as e:
                print(f"\nErro ao gerar a resposta: {e}")
                
                # Tentativa alternativa com configurações ainda mais conservadoras
                try:
                    print("\nTentando novamente com configurações alternativas (sem streaming)...")
                    output = llm.create_completion(
                        prompt=prompt,
                        max_tokens=8192,  # Ainda alto, mas um pouco menor para maior probabilidade de sucesso
                        temperature=0.2,
                        top_p=0.8,
                        repeat_penalty=1.05,
                        stop=["### Instrução:", "### RESPOSTA:"],
                        stream=False  # Fallback para modo não-streaming em caso de falha
                    )
                    response = output['choices'][0]['text']
                    print(response)
                    full_response = response
                except Exception as e2:
                    print(f"\nSegunda tentativa também falhou: {e2}")
                    continue
            
            # Salva a interação no log apenas se houver uma resposta válida
            if full_response.strip():
                save_to_log(user_input, full_response)
            else:
                print("Não foi possível gerar uma resposta completa. Verifique o system prompt e tente novamente.")
            
            # Pergunta se o usuário quer limpar o arquivo de input para a próxima pergunta
            clear = input("\nLimpar arquivo de input para próxima consulta? (s/n): ")
            if clear.lower() in ['s', 'sim', 'y', 'yes']:
                with open("input.yaml", "w", encoding="utf-8") as f:
                    f.write("input: |\n  ")
                print("Arquivo input.yaml limpo. Adicione sua próxima pergunta nele.")
        except Exception as e:
            print(f"Erro durante a execução: {e}")
            continue

if __name__ == "__main__":
    main() 