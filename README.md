## Projeto Detector de COVID-19

Este projeto é baseado em *Machine Learning* e na linguagem de programação python. A proposta é realizar a detecção de COVID-19 em imagens de raio-x, dentro do contexto pandemico ocorrido entre o ano de 2019-2022. A base de dados selececionada pode ser adquirida neste [Link](https://github.com/ieee8023/covid-chestxray-dataset). A proposta é gerar um modelo de predição para disitinguir entre uma imagem com COVID-19 e sem COVID-19.

### Imagens da Base de Dados
![Screenshot](/img/02.png)

## Execução do Projeto
```bash
    tensorflow = 2.4
    opencv-python = 4.6.0
    python = 3.8.8
    numpy = 1.21
```

A arquitetura para treino e geração do modelo é a VGG-16, metodo de otimização Adam e treinamento em 100 epocas.

## Resultados do Detector

![Screenshot](/img/Image_detect.png)


