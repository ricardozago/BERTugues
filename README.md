# BERTugues 🇧🇷

![BERTugues](https://img.shields.io/badge/Model-BERTugues-blue) ![Language](https://img.shields.io/badge/Language-Portuguese-green) ![Architecture](https://img.shields.io/badge/Architecture-BERT-orange)

**BERTugues** é um modelo de linguagem focado no processamento da língua portuguesa (PT-BR). O modelo foi desenhado e treinado para ser uma alternativa eficiente aos modelos multilingues e aos modelos focados no português já existentes, apresentando um **tokenizador altamente otimizado** que evita a fragmentação excessiva de palavras, resultando em melhor compressão de contexto e eficiência durante o fine-tuning e a inferência.

---

## 🚀 Por que usar o BERTugues?

1. **Tokenização Eficiente:** Em comparações diretas no dataset ASSIN 2, o tokenizador do BERTugues apresenta uma taxa significativamente menor de fragmentação de sub-palavras (com redução dos fragmentos iniciados em `##`) em comparação ao `mBERT` e ao `BERTimbau`. Isso significa que o modelo precisa de menos tokens para representar as mesmas frases, aumentando o "espaço real" de contexto e diminuindo o custo computacional.
2. **Ausência de Viés de Vocabulário Estrangeiro:** O vocabulário foi limpo para remover caracteres asiáticos e símbolos não utilizados no português.
3. **Versatilidade:** Ideal para Fine-Tuning de Classificação de Texto, RAG (Retrieval-Augmented Generation), Reconhecimento de Entidades Nomeadas (NER) e demais tarefas de NLP.

---

## � Notebooks do Repositório

O repositório conta com uma série de Jupyter Notebooks focados em demonstrar as capacidades e a avaliação do **BERTugues**:

- [**01 Tokenizador comparação.ipynb**](https://github.com/ricardozago/BERTugues/blob/main/01%20Tokenizador%20compara%C3%A7%C3%A3o.ipynb): Análise comparativa entre os tokenizadores do mBERT, BERTimbau e BERTugues, focando na menor fragmentação de palavras (subtokens).
- [**02 Benchmark Modelos.ipynb**](https://github.com/ricardozago/BERTugues/blob/main/02%20Benchmark%20Modelos.ipynb): Benchmark avaliando e comparando extração de features textuais (Random Forest) entre BERTugues, BERTimbau Base, BERTimbau Large e mBERT.
- [**03 Classificação com Random Forest.ipynb**](https://github.com/ricardozago/BERTugues/blob/main/03%20Classifica%C3%A7%C3%A3o%20com%20Random%20Forest.ipynb): Extração de embeddings do texto usando estratégias `CLS` e `Mean` Pooling e classificação de sentimentos via scikit-learn.
- [**04 Classificação com Fine-Tuning.ipynb**](https://github.com/ricardozago/BERTugues/blob/main/04%20Classifica%C3%A7%C3%A3o%20com%20Fine-Tuning.ipynb): Exemplo completo de Fine-Tuning do modelo com arquitetura customizada lidando com os diferentes formatos de Pooling para classificação.
- [**05 RAG Simplificado (Retrieval).ipynb**](https://github.com/ricardozago/BERTugues/blob/main/05%20RAG%20Simplificado%20(Retrieval).ipynb): Construção de um buscador semântico eficiente utilizando Mean Pooling e distância de cosseno, base para aplicações RAG.
- [**06 MLM Previsão de Palavras.ipynb**](https://github.com/ricardozago/BERTugues/blob/main/06%20MLM%20Previs%C3%A3o%20de%20Palavras.ipynb): Demonstração do funcionamento nativo (*Masked Language Modeling*) usando o pipeline `fill-mask` para dedução contextual em português.
- [**07 Reconhecimento de Entidades Nomeadas (NER).ipynb**](https://github.com/ricardozago/BERTugues/blob/main/07%20Reconhecimento%20de%20Entidades%20Nomeadas%20(NER).ipynb): Exemplo de Fine-Tuning na tarefa de Token Classification (NER) sob textos jurídicos, utilizando o dataset **LeNER-Br**.

---

## �💻 Guia de Uso e Exemplos

Você pode utilizar o BERTugues facilmente através da biblioteca `transformers` da Hugging Face.

### 1. Previsão de Palavras Ocultas (Masked LM)
A maneira mais rápida de testar o entendimento contextual da língua portuguesa.

```python
from transformers import pipeline

unmasker = pipeline("fill-mask", model="ricardoz/BERTugues-base-portuguese-cased")

frase = "Eu gosto de comer arroz com [MASK]."
resultados = unmasker(frase, top_k=3)

for res in resultados:
    print(f"{res['sequence']} (Score: {res['score']:.4f})")
# Saída esperada: "Eu gosto de comer arroz com feijão." etc...
```

### 2. Extração de Embeddings (Feature Extraction)
Use o modelo como um codificador semântico puro para treinar modelos tradicionais como o Random Forest no Scikit-Learn ou sistemas de RAG.

```python
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "ricardoz/BERTugues-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

texto = "O produto é excelente, recomendo a todos!"
inputs = tokenizer(texto, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Extração com estratégia CLS (1º token)
cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()

# CLS pronto para ser usado como X_train em um Random Forest, SVM, Regressão Logística, etc.
```

### 3. RAG e Busca Semântica (Retrieval)
Aproveite o **Mean Pooling** para representar frases completas.

```python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

def get_sentence_embedding(texto, model, tokenizer):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean Pooling
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    
    sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).numpy()

# Comparando perguntas com base de conhecimento (Vector Database)
pergunta_emb = get_sentence_embedding("Como fazer um bolo?", model, tokenizer)
documento_emb = get_sentence_embedding("Você precisa de farinha, ovos, açúcar e cenoura.", model, tokenizer)

similaridade = cosine_similarity(pergunta_emb, documento_emb)
print(f"Score de Similaridade: {similaridade[0][0]:.4f}")
```

### 4. Fine-Tuning para Classificação de Texto
Ajuste os pesos do modelo diretamente para receber predições de ponta a ponta.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "ricardoz/BERTugues-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) # Ex: Positivo / Negativo

inputs = tokenizer("A qualidade deste aparelho celular me decepcionou muito.", return_tensors="pt")
logits = model(**inputs).logits

predicao = logits.argmax(dim=-1).item()
print("Classe predita:", predicao)
```

### 5. Reconhecimento de Entidades Nomeadas (NER)
Pode ser adaptado perfeitamente para identificação de PESSOA, ORGANIZAÇÃO, LEGISLAÇÃO, aplicando-se sobre bases como o **LeNER-Br** (Legal Named Entity Recognition).

```python
from transformers import AutoModelForTokenClassification

# Instanciar o modelo pronto para Token Classification
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    num_labels=13 # Ajuste de acordo com a quantidade de tags I-O-B do seu dataset
)
```

## 🛠️ Requisitos e Instalação

```bash
pip install transformers torch
```

## 🤝 Contribuições

Este repositório contém os notebooks de exemplo, validação, criação do DataLoader e Fine-tuning usados na validação das capacidades do BERTugues. Fique à vontade para submeter *pull-requests* melhorando ou expandindo os benchmarks!

Escrito com ajuda de AI! ✨
