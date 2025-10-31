# 🎥 Detecção Automática de Logotipos em Vídeos — Nike Swoosh Example

**Autores:**

* Geisbelly Victória Moraes
* Lucas Bueno
* Samuel Matsukami Cruz Kagueiama

**Disciplina:** Computação Gráfica — ULBRA Palmas

---

## 🧩 Descrição do Projeto

Este projeto implementa um **sistema de detecção automática de logotipos em vídeos**, com foco em **logos simples** (exemplo: o *swoosh* da Nike), utilizando técnicas de **Visão Computacional**.

O código identifica **quando e onde** a logo aparece no vídeo, calcula o **tempo de exposição**, o **percentual da área ocupada na tela**, e gera **imagens e mapas de calor** da presença da marca.

> 💡 Este trabalho é uma **reprodução simplificada e otimizada** do conceito apresentado no projeto acadêmico
> *“Reconhecimento e Extração de Marcas em Vídeos de Partidas de Futebol” (Dell/Insper, 2022).*

---

## ⚙️ Funcionalidades Principais

✅ **Detecção automática da logo** em vídeos
✅ **Rastreamento rápido entre frames** usando *MOSSE Tracker*
✅ **Re-detecção periódica** via *Template Matching* em bordas (*Canny*)
✅ Combina **precisão** (*template matching* multi-escala e multi-rotação)
com **velocidade** (rastreamento entre detecções)

---

## 📊 Gera automaticamente:

* 🕒 **Tempo total de exposição** da logo no vídeo
* 📏 **Percentual médio da área ocupada** na tela
* 🖼️ **Frames anotados** com a logo detectada
* 🔥 **Mapa de calor (heatmap)** e **heatmap sobreposto** ao vídeo

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia                         | Função                              |
| ---------------------------------- | ----------------------------------- |
| **Python 3.10+**                   | Linguagem principal                 |
| **OpenCV (opencv-contrib-python)** | Processamento e visão computacional |
| **NumPy**                          | Manipulação numérica e de matrizes  |

---

## 📦 Instalação

No terminal, execute:

```bash
py -m pip install opencv-contrib-python numpy
```

---

## ▶️ Uso

Coloque na **mesma pasta** os seguintes arquivos:

```
detector_logo_video.py
meu_video.mp4
logo_template.png
```

**Onde:**

* `detector_logo_video.py` → script principal
* `meu_video.mp4` → vídeo onde a logo será detectada
* `logo_template.png` → imagem da logo (ex: swoosh branco em fundo preto)

---

### 🔧 Execução

```bash
py detector_logo_video.py
```

---

## 📁 Saídas Geradas

São salvas automaticamente na pasta `saida_detect/`:

| Arquivo                      | Descrição                                                              |
| ---------------------------- | ---------------------------------------------------------------------- |
| `frame_detectado_XXXXXX.jpg` | Frames com a logo destacada (até o limite definido em `DRAW_SAMPLE_N`) |
| `heatmap.png`                | Mapa de calor normalizado da presença da logo                          |
| `heatmap_sobreposto.png`     | Heatmap sobreposto a um frame de referência                            |

---

## 📈 Funcionamento Interno

1. **Pré-processamento:** o vídeo é convertido em tons de cinza e suavizado.
2. **Template Matching:** busca inicial pela logo em diferentes escalas e rotações.
3. **Rastreamento (MOSSE):** acompanha a posição da logo entre frames consecutivos.
4. **Revalidação periódica:** a cada N frames, a logo é rechecada para corrigir drift.
5. **Cálculo final:** tempo de exposição, área média e mapa de calor.

---

## 📚 Referência

Sousa, B. C., & Azevedo, T. H. (2022).
*Reconhecimento e Extração de Marcas em Vídeos de Partidas de Futebol.*
Trabalho de Conclusão de Curso — Insper Instituto de Ensino e Pesquisa.
[Disponível aqui](https://repositorio-api.insper.edu.br/server/api/core/bitstreams/12f7810a-e2f8-4add-a185-d17b12e5d999/content)

---

## 🧠 Conclusão

O projeto demonstra, de forma prática, como técnicas de **visão computacional** podem ser aplicadas em **automação de mídia** e **análise publicitária esportiva**, permitindo medir o impacto visual de marcas em vídeos de forma eficiente.
