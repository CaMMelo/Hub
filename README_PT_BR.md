 <p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/logo/logo-explainer-bg.png" width="50%"/>
    </br>
</p>

<p align="center">
    <a href="http://docs.activeloop.ai/">
        <img alt="Docs" src="https://readthedocs.org/projects/hubdb/badge/?version=latest">
    </a>
    <a href="https://pypi.org/project/hub/"><img src="https://badge.fury.io/py/hub.svg" alt="PyPI version" height="18"></a>
    <a href="https://pypi.org/project/hub/"><img src="https://img.shields.io/pypi/dm/hub.svg" alt="PyPI version" height="18"></a>
    <a href="https://app.circleci.com/pipelines/github/activeloopai/Hub">
    <img alt="CircleCI" src="https://img.shields.io/circleci/build/github/activeloopai/Hub?logo=circleci"> </a>
     <a href="https://github.com/activeloopai/Hub/issues">
    <img alt="GitHub issues" src="https://img.shields.io/github/issues/activeloopai/Hub"> </a>
    <a href="https://codecov.io/gh/activeloopai/Hub/branch/master"><img src="https://codecov.io/gh/activeloopai/Hub/branch/master/graph/badge.svg" alt="codecov" height="18"></a>
    <a href="https://twitter.com/intent/tweet?text=The%20fastest%20way%20to%20access%20and%20manage%20PyTorch%20and%20Tensorflow%20datasets%20is%20open-source&url=https://activeloop.ai/&via=activeloopai&hashtags=opensource,pytorch,tensorflow,data,datascience,datapipelines,activeloop,dockerhubfordatasets"> 
        <img alt="tweet" src="https://img.shields.io/twitter/url/http/shields.io.svg?style=social"> </a>  
   </br> 
    <a href="https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ">
  <img src="https://user-images.githubusercontent.com/13848158/97266254-9532b000-1841-11eb-8b06-ed73e99c2e5f.png" height="35" /> </a>

---

</a>
</p>

<h3 align="center"> Introducing Data 2.0, powered by Hub. </br>The fastest way to store, access & manage datasets with version-control for PyTorch/TensorFlow. Scalable data pipelines.</h3>

---

[ [English](./README.md) | [简体中文](./README_CN.md) | Português(BR)]

### Pra que serve o Hub?

Software 2.0 precisa de Dados 2.0, e o Hub fornece isso. Na maior parte do tempo os Cientistas de Dados/Pesquisadores de Aprendizagem de Maquina trabalham no preprocessamento e gestão de dados ao invés do treinamento de modelos. Com o Hub, iremos dar um fim a isso. Armazenamos os seus datasets em nuvem (até mesmo em escalas de petabytes) como se fossem arrays numpy, dessa forma você pode acessar e trabalhar com eles de qualquer lugar. O Hub faz com que qualquer tipo de dado (imagem, texto, audio ou video) armazenados em nuvem possa ser utilizado como se estivessem armazenado localmente. Com a mesma visão dos dados, seu time sempre estará em sincronia.

O Hub está sendo usado pela Waymo, Red Cross, World Resources Institute, Omdena, e outros.

### Features 

* Armazena e recupera grandes datasets com versionamento
* Colaborativo igual ao Google Docs: Vários cientistas de dados trabalham nos mesmos dados em sincronia e sem interrupções
* Acesso de vários locais simultaneamente
* Deploy em qualquer lugar - localmente, no Google Cloud, S3, Azure e também no Activeloop (por padrão - e de graça)
* Integração com suas ferramentas de aprendizagem de máquina favoritas como Numpy, Dask, Ray, [PyTorch](https://docs.activeloop.ai/en/latest/integrations/pytorch.html), ou [TensorFlow](https://docs.activeloop.ai/en/latest/integrations/tensorflow.html)
* Crie arrays tão grandes o quanto quiser. Você pode salvar imagens de até 100k por 100k!
* Mantenho o formato de cada amostra dinâmico. Dessa forma você pode armazenar arrays grandes e pequenos em um único array.
* [Visualize](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme) qualquer parte dos dados em uma questão de segundos sem manipulações redundantes

## Getting Started

### Acesso a dados publicos. RAPIDO

Para acessar um dataset public, seriam necessárias dezenas de linhas de código e muitas horas para ler e entender a API além do download dos dados. Com o Hub, você só precisa de duas linhas, e **pode começar a trabalhar no dataset em menos de 3 minutos**.

```sh
pip3 install hub
```

Acessar datasets publicos no Hub seguindo uma maneira direta que precisa somente de poucas linhas de código bem simples. Execute esse trecho para carregar as primeiras mil imagens do [dataset MNIST](https://app.activeloop.ai/dataset/activeloop/mnist/?utm_source=github&utm_medium=repo&utm_campaign=readme) no formato de um array numpy:
```python
from hub import Dataset

mnist = Dataset("activeloop/mnist")  # carregamento "preguiçoso" dos dados MNIST
# economia de tempo com *compute* carregando somente os dados necessários
mnist["image"][0:1000].compute()
```

Voce pode encontrar outros datasets populares em [app.activeloop.ai](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme).

### Treine um modelo

Carregue os dados e treine seu model **diretamente**. O Hub é integrado com o PyTorch e o TensorFlow e faz a conversão entre os formatos de uma maneira simples. Veja um exemplo utilizando PyTorch:

```python
from hub import Dataset
import torch

mnist = Dataset("activeloop/mnist")
# conversão do MNIST para o formato do PyTorch
mnist = mnist.to_pytorch(lambda x: (x["image"], x["label"]))

train_loader = torch.utils.data.DataLoader(mnist, batch_size=1, num_workers=0)

for image, label in train_loader:
    # Rotina de treinamento
```

### Crie um dataset local 
Se preferir trabalhar com seus proprios dados localmente, voce pode criar seus próprio dataset:
```python
from hub import Dataset, schema
import numpy as np

ds = Dataset(
    "./data/dataset_name",  # caminho do arquivo do dataset
    shape = (4,),  # seguindo a convensão de formato do numpy
    mode = "w+",  # modo de leitura e escrita
    schema = {  # campos nomeados de dados que podem especificar tipos
    # Tensor é uma estrutura genérica que contém qualquer tipo de dado
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    }
)

# preenche os campos de dados com valores (inicialização com zeros)
ds["image"][:] = np.zeros((4, 512, 512))
ds["label"][:] = np.zeros((4, 512, 512))
ds.commit()  # executa a criação do dataset
```

Também é possível especificar `s3://bucket/path`, `gcs://bucket/path` ou caminhos da azure. [Esse link](https://docs.activeloop.ai/en/latest/simple.html#data-storage) contém mais informações sobre armazenamento em nuvem. Se precisar de um dataset publico que não encontrar no Hub, você pode [enviar um pedido](https://github.com/activeloopai/Hub/issues/new?assignees=&labels=i%3A+enhancement%2C+i%3A+needs+triage&template=feature_request.md&title=[FEATURE]+New+Dataset+Required%3A+%2Adataset_name%2A). Disponibilizaremos o quanto antes!

### Envie seu dataset e acesse ele de <ins>qualquer lugar</ins> em 3 etapas:

1. Crie uma conta em [Activeloop](https://app.activeloop.ai/register/?utm_source=github&utm_medium=repo&utm_campaign=readme) e faça o login localmente:
```sh
hub register
hub login

# Se preferir, adicione seu nome de usuario e senha com os argumentos (para plataformas como Kaggle)
hub login -u username -p password
```

2. Crie seu dataset, especificando o nome e envie para sua conta. Por exemplo:
```python
from hub import Dataset, schema
import numpy as np

ds = Dataset(
    "username/dataset_name",
    shape = (4,),
    mode = "w+",
    schema = {
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    }
)

ds["image"][:] = np.zeros((4, 512, 512))
ds["label"][:] = np.zeros((4, 512, 512))
ds.commit()
```

3. Acesse de qualquer lugar do mundo, em qualquer aparelho com o seguinte comando:
```python
from hub import Dataset

ds = Dataset("username/dataset_name")
```


## Documentação

For more advanced data pipelines like uploading large datasets or applying many transformations, please refer to our [documentation](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme).

## Tutorial Notebooks
The [examples](https://github.com/activeloopai/Hub/tree/master/examples) directory has a series of examples and notebooks giving an overview of Hub. Some of the notebooks are listed of below.

| Notebook  	|   Description	|   	|
|:---	|:---	|---:	|
| [Uploading Images](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) | Overview on how to upload and store images on Hub |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) |
| [Uploading Dataframes](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	| Overview on how to upload Dataframes on Hub  	| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	|
| [Uploading Audio](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) | Explains how to handle audio data in Hub|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) |
| [Retrieving Remote Data](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) | Explains how to retrieve Data| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) |
| [Transforming Data](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) | Briefs on how data transformation with Hub|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) |
| [Dynamic Tensors](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) | Handling data with variable shape and sizes|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) |
| [NLP using Hub](https://github.com/activeloopai/Hub/blob/master/examples/nlp_using_hub.ipynb) | Fine Tuning Bert for CoLA|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/nlp_using_hub.ipynb) |


## Use Cases
* **Satellite and drone imagery**: [Smarter farming with scalable aerial pipelines](https://activeloop.ai/usecase/intelinair?utm_source=github&utm_medium=repo&utm_campaign=readme), [Mapping Economic Well-being in India](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005), [Fighting desert Locust in Kenya with Red Cross](https://omdena.com/projects/ai-desert-locust/)
* **Medical Images**: Volumetric images such as MRI or Xray
* **Self-Driving Cars**: [Radar, 3D LIDAR, Point Cloud, Semantic Segmentation, Video Objects](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
* **Retail**: Self-checkout datasets
* **Media**: Images, Video, Audio storage

## Community

Join our [**Slack community**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) to get help from Activeloop team and other users, as well as stay up-to-date on dataset management/preprocessing best practices.

<img alt="tweet" src="https://img.shields.io/twitter/follow/activeloopai?label=stay%20in%20the%20Loop&style=social"> on Twitter.

As always, thanks to our amazing contributors!    

<a href="https://github.com/activeloopai/hub/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=activeloopai/hub" />
</a>

Made with [contributors-img](https://contrib.rocks).

Please read [CONTRIBUTING.md](CONTRIBUTING.md) to know how to get started with making contributions to Hub.

## Examples
Activeloop's Hub format lets you achieve faster inference at a lower cost. We have 30+ popular datasets already on our platform. These include:
- COCO
- CIFAR-10
- PASCAL VOC
- Cars196
- KITTI
- EuroSAT 
- Caltech-UCSD Birds 200
- Food101

Check these and many more popular datasets on our [visualizer web app](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme) and load them directly for model training!

## README Badge

Using Hub? Add a README badge to let everyone know: 


[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

```
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)
```

## Disclaimers

Similarly to other dataset management packages, `Hub` is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/activeloopai/Hub/issues/new). Thanks for your contribution to the ML community!


## Acknowledgement
 This technology was inspired from our experience at Princeton University and would like to thank William Silversmith @SeungLab with his awesome [cloud-volume](https://github.com/seung-lab/cloud-volume) tool. We are heavy users of [Zarr](https://zarr.readthedocs.io/en/stable/) and would like to thank their community for building such a great fundamental block. 
