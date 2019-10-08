# Classificador SVM

O classificador pode ser utilizado para classificar imagens em n classes, inicialmente criado para classificação de imagens dia / noite.

  - Realiza treinamento
  - Gera modelo
  - Classfica e calcula o acerto.

### Como utilizar

necessario algumas libs: sklearn, opencv, numpy, os, argv e _pickle

Install the dependencies and devDependencies and start the server.

```sh
$ pip3 install sklearn opencv numpy os argv _pickle
```

Depois de instalado as dependencias

para treino:
```sh
$ python3 svm.py --train [img_folder]
```
para classificação:
```sh
$ python3 svm.py --classify [img_folder]
```
