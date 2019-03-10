# 第1步：了解Rasa技术栈
Rasa Stack是一套开源机器学习工具

- NLU   = 用于自然语言理解的库，具有意图分类和实体提取
- Core  = 具有基于机器学习的对话管理的聊天机器人框架

NLU和Core是独立的，可以单独使用。

**NLU** 根据您之前的培训数据了解用户的消息：

- 意图分类：根据预定义的意图解释意义
- 实体提取：识别结构化数据

**Core** 决定此对话中接下来会发生什么。它基于机器学习的对话管理根据NLU的输入，对话历史和您的训练数据预测下一个最佳动作。

# 第2步：基本工作过程
使用Rasa NLU教授机器人以了解用户输入

1. 创建NLU示例
2. 定义NLU模型配置
3. 训练NLU模型
4. 测试模型

教授机器人使用Rasa Core进行响应

5. 写故事
6. 定义域
7. 训练对话模型

## 1. 创建NLU示例

第一件事是定义机器人应该理解的用户消息。您将通过定义意图并提供用户可能表达的几种方式来实现此目的。

```markdown
## intent:greet
- hey
- hello
- hi
- good morning
- good evening
- hey there

## intent:goodbye
- bye
- goodbye
- see you around
- see you later
```

## 2. 定义NLU模型配置

NLU模型配置定义了如何训练NLU模型以及如何提取文本输入中的特征。

```yml
language: en
pipeline: tensorflow_embedding
```

## 3. 训练NLU模型

## 4. 测试模型

## 5. 写故事

在此阶段，您将教您的聊天机器人使用Rasa Core回复您的消息。Rasa Core将训练对话管理模型，并预测机器人应如何在特定的对话状态下做出响应。

Rasa Core模型以训练“故事”的形式从真实的会话数据中学习。故事是用户和机器人之间的真实对话，其中用户输入表示为意图，机器人的响应表示为动作名称。下面是一个简单对话的例子：用户向我们的机器人问好，然后机器人问好。这就是它看起来像一个故事：

```markdown
## happy path
* greet
  - utter_greet
* mood_great
  - utter_happy

## say goodbye
* goodbye
  - utter_goodbye
```

## 6. 定义域
```yml
intents:
  - greet
  - goodbye
  - mood_affirm
  - mood_deny
  - mood_great
  - mood_unhappy

actions:
- utter_greet
- utter_cheer_up
- utter_did_that_help
- utter_happy
- utter_goodbye

templates:
  utter_greet:
  - text: "Hey! How are you?"

  utter_cheer_up:
  - text: "Here is something to cheer you up:"
    image: "https://i.imgur.com/nGF1K8f.jpg"

  utter_did_that_help:
  - text: "Did that help you?"

  utter_happy:
  - text: "Great carry on!"

  utter_goodbye:
  - text: "Bye"
```
intents : 你希望用户说的话。见 Rasa NLU

actions	: 机器人可以做的事情

templates : 你的机器人可以说的东西的模板字符串

## 7. 训练对话模型

# 第3步：NLU用户指南

## 选择Rasa NLU Pipeline
### 二选一
如果您的训练示例少于1000个，并且您的语言有spaCy模型，请使用spacy_sklearn管道。

如果您有1000个或更多标记的话语，请使用tensorflow_embedding管道。

两条最重要的管道是tensorflow_embedding和spacy_sklearn。它们之间最大的区别是spacy_sklearn管道使用来自GloVe或fastText的预先训练的单词向量。相反，tensorflow_embedding管道不使用任何预先训练的单词向量，而是专门为您的数据集拟合这些。

### 理解Rasa NLU流水线
在Rasa NLU中，传入消息由一系列组件处理。这些组件在所谓的处理流水线中一个接一个地执行。有实体提取，意图分类，预处理等组件。如果要添加自己的组件，例如运行拼写检查或进行情绪分析，请使用自定义组件。

每个组件处理输入并创建输出。输出可以由管道中此组件之后的任何组件使用。有些组件只生成管道中其他组件使用的信息，有些组件将在管道处理完成后返回 *Output* 属性。

## 语言支持
*tensorflow_embedding* 管道可用于任何语言。

其他管道有一些限制，并支持那些具有预训练单词向量的语言。

### 预先训练的单词向量
使用spaCy后端，可以加载fastText向量，这些向量可用于数百种语言。

## 实体提取
|Component	|Requires	|Model	|notes
| ------------- |:------:|:-------:| -----:|
|ner_crf	|sklearn-crfsuite	|conditional random field	|good for training custom entities
|ner_spacy	|spaCy	|averaged perceptron	|provides pre-trained entities
|ner_duckling_http	|running duckling	|context-free grammar	|provides pre-trained entities
|ner_mitie	|MITIE	|structured SVM	|good for training custom entities

```json
{
  "text": "show me chinese restaurants",
  "intent": "restaurant_search",
  "entities": [
    {
      "start": 8,
      "end": 15,
      "value": "chinese",
      "entity": "cuisine",
      "extractor": "ner_crf",
      "confidence": 0.854,
      "processors": []
    }
  ]
}
```

## 评估和改进模型

### 从反馈中改进模型

一旦有一个版本的模型在运行，Rasa NLU服务器会将对每个请求记录到文件中。

```json
{
  "user_input":{
    "entities":[]   ],
    "intent":{
      "confidence":0.32584617693743012,
      "name":"restaurant_search"
    },
    "text":"nice thai places",
    "intent_ranking":[ ... ]
  },
  ...
  "model":"default",
  "log_time":1504092543.036279
}
```
用户所说的内容是改进模型的最佳培训数据来源。当然，您的模型并不完美，因此您必须手动完成每个预测并纠正任何错误，然后再将其添加到您的训练数据中。如上例，实体'泰国'没有被当作菜肴。

### 评估模型

Rasa NLU有一种 *evaluate* 模式可以帮助您评估模型。机器学习中的标准技术是将一些数据作为测试集分开。

如果您没有单独的测试集，您仍然可以使用交叉验证来估计模型的优化程度。

## 置信度和回退意图

您可以使用置信度分数选择何时忽略Rasa NLU的预测和触发回退行为，例如要求用户重新措辞。

选择置信度截止值的一种好方法是计算模型对测试集的置信度，并比较正确和错误预测的示例的置信度值。

始终记住，置信度得分不是预测正确的真实概率，它只是由模型定义的度量，它大致描述了您的输入与训练数据的相似程度。


# 第4步：CORE用户指南

## 架构

此图显示了Rasa Core应用程序如何响应消息的基本步骤：

![High-Level Architecture](https://rasa.com/docs/core/_images/rasa_arch_colour.png)

步骤是：

1. 消息被接收并传递给 Interpreter，它将其转换为包含原始文本，意图和找到的任何实体的字典。

2. Tracker 是跟踪通话状态的对象。它接收新消息进入的信息。

3. 策略接收跟踪器的当前状态。

4. 该策略选择接下来采取的行动。

5. 选择的操作由跟踪器记录。

6. 响应被发送给用户。

## 动作

动作是机器人响应用户输入而执行的内容。Rasa Core中有三种动作：

1. 默认动作（*action_listen，action_restart， action_default_fallback*）
2. 说话动作，以 *utter_* 开始，只是向用户发送消息
3. 自定义动作，任何其他动作，这些动作可以运行任意代码

### 说话动作
要定义一个说话动作，请在域文件中添加一个话语模板，该模板以 *utter_* 开头：

```yml
templates:
  utter_my_message:
    - "this is what I want my action to say!"
```

### 自定义动作
操作可以运行您想要的任何代码。自定义动作可以打开灯光，向日历添加事件，检查用户的银行余额或您可以想象的任何其他内容。

当预测到自定义操作时，Core将调用您指定的端点。此端点应该是响应此调用的Web服务，运行代码并可选地返回信息以修改对话状态。

你可以用node.js，.NET，java或任何其他语言创建一个动作服务器，并在那里定义你的动作。但是 **rasa_core_sdk** 提供了一个小的python sdk来使开发变得更加容易。

### 默认动作
有三种默认操作：

action_listen : 停止预测更多动作并等待用户输入

action_restart : 重置整个会话

action_default_fallback : 撤消最后一条用户消息（就好像用户没有发送它）并发出机器人不理解的消息。

## 使用插槽



## 填槽



## 机器人的响应



## 互动学习



## 回退动作



## 训练和策略



## 调试



## 评估和测试


