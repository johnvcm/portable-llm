# Projeto LLM Local com DeepSeek

Este projeto fornece um ambiente para execução local de modelos de linguagem (LLM) da família DeepSeek.

## Estrutura do Projeto

### Diretórios Principais

- `logs/` - Armazena histórico de interações com o modelo
- `models/` - Contém os modelos LLM baixados
- `.venv/` - Ambiente virtual Python criado pelo `uv`
- `.cache/` - Cache para downloads da Hugging Face

### Scripts Principais

- `app.py` - Aplicativo principal usando o modelo DeepSeek Coder 6.7B
- `app_for_deepseek_v2.py` - Versão usando o modelo DeepSeek Coder V2 Lite
- `download_model.py` - Baixa o modelo DeepSeek Coder 6.7B
- `download_deepseek_v2_lite.py` - Baixa o modelo DeepSeek Coder V2 Lite

### Arquivos de Configuração

- `system_prompt.yaml` - Prompt de sistema para guiar o modelo
- `input.yaml` - Arquivo para inserir perguntas para o modelo processar

## Como Executar o Ambiente LLM

### 1. Instalação do UV

Primeiro, instale o `uv` no seu sistema. O `uv` é um instalador e gerenciador de pacotes Python rápido.

```bash
# Exemplo de instalação (verifique a documentação oficial para o seu sistema)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Configuração do Ambiente Virtual e Instalação de Dependências

Com o `uv` instalado, você pode criar o ambiente virtual e instalar as dependências do projeto com um único comando:

```bash
uv sync
```

Este comando irá ler o arquivo `pyproject.toml`, criar um ambiente virtual (`.venv`) e instalar todas as dependências necessárias.

Para ativar o ambiente virtual, use:

```bash
source .venv/bin/activate
```

### 3. Baixar Modelos

```bash
python download_model.py
```

ou

```bash
python downloads/download_deepseek_v2_lite.py
```

### 4. Executar o Aplicativo

1. Escreva sua consulta no arquivo `input.yaml`
2. Execute um dos aplicativos:

```bash
python app.py
```

ou

```bash
python app_for_deepseek_v2.py
```

3. O sistema carregará o modelo, processará sua consulta e salvará a interação na pasta `logs/`

## Troubleshooting

### LLM-Related Issues

- Se o modelo não carregar, verifique se há espaço suficiente no disco e RAM disponível
- Os modelos são grandes, certifique-se de que o download foi concluído corretamente
- Ajuste os parâmetros de contexto (`n_ctx`) nos arquivos `app.py` ou `app_for_deepseek_v2.py` se tiver problemas de memória

## Desativando o Ambiente Virtual

Quando terminar, para desativar o ambiente virtual, execute:

```bash
deactivate
```
