Este projeto implementa um sistema de detecção automática de logotipos em vídeos — com foco em logos simples como o swoosh da Nike — utilizando técnicas de Visão Computacional.

O código identifica quando e onde a logo aparece no vídeo, calcula tempo de exposição, percentual de área ocupada na tela e gera imagens e mapas de calor da presença da marca.

Este trabalho é uma reprodução simplificada e otimizada do conceito apresentado no projeto acadêmico “Reconhecimento e Extração de Marcas em Vídeos de Partidas de Futebol (Dell/Insper, 2022)”.


Funcionalidades:

Detecção automática da logo em vídeo.

Rastreamento rápido entre frames usando MOSSE Tracker.

Re-detecção periódica via Template Matching em bordas (Canny).

Combina precisão (template matching multi-escala e multi-rotação) e velocidade (rastreamento entre detecções).

Gera:

Tempo total de exposição da logo no vídeo;

Percentual médio da área ocupada na tela;

Frames anotados com a logo detectada;

Heatmap e heatmap sobreposto ao vídeo.

Tecnologias Utilizadas:

Python 3.10+

OpenCV (opencv-contrib-python) — visão computacional

NumPy — manipulação numérica e de matrizes


Instale as dependências:

py -m pip install opencv-contrib-python numpy


Uso:

Coloque na mesma pasta:

detector_logo_video.py

meu_video.mp4 (o seu vídeo)

logo_template.png (a imagem da logo — ex.: swoosh branco em fundo preto)

Execute o script:

py detector_logo_video.py


Saídas:

Geradas em saida_detect/:

frame_detectado_XXXXXX.jpg → frame com a logo destacada (até DRAW_SAMPLE_N unidades).

heatmap.png → mapa de calor normalizado da presença da logo.

heatmap_sobreposto.png → heatmap sobreposto a um frame de referência.
