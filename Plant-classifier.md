
**Plant Image Classifier** using **Amazon SageMaker real-time endpoint** — the most ML-native, managed option.

---

## 🌱 Project Name: `plant-classifier-sagemaker`

### 📁 Project Structure:

```
plant-classifier-sagemaker/
├── model/
│   └── plant_model_scripted.pt         ← TorchScript model
├── sagemaker/
│   ├── inference.py                    ← Inference script (model_fn, input_fn, etc.)
│   └── model.tar.gz                    ← Package to upload to S3
├── deploy/
│   └── deploy.py                       ← Script to create SageMaker model and endpoint
├── test/
│   └── invoke_endpoint.py              ← Test the endpoint with a sample image
├── requirements.txt                    ← Python dependencies
└── README.md
```

---

### 🔧 1. `inference.py`

```python
# sagemaker/inference.py
import torch
from torchvision import transforms
from PIL import Image
import io

labels = ['daisy', 'rose', 'sunflower']

def model_fn(model_dir):
    model = torch.jit.load(f"{model_dir}/plant_model_scripted.pt")
    model.eval()
    return model

def input_fn(request_body, content_type):
    image = Image.open(io.BytesIO(request_body)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def predict_fn(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs, 1)
    return labels[predicted.item()]

def output_fn(prediction, content_type):
    return prediction
```

---

### 📦 2. Package Model

```bash
cd sagemaker/
tar -czvf model.tar.gz inference.py ../model/plant_model_scripted.pt
```

Upload to S3:

```bash
aws s3 cp model.tar.gz s3://<your-s3-bucket>/plant-classifier/
```

---

### 🚀 3. `deploy.py`

```python
# deploy/deploy.py
import sagemaker
from sagemaker.pytorch import PyTorchModel

role = '<your-sagemaker-execution-role>'
model_path = 's3://<your-s3-bucket>/plant-classifier/model.tar.gz'

model = PyTorchModel(
    entry_point='inference.py',
    model_data=model_path,
    role=role,
    framework_version='1.12.0',
    source_dir='.',
    py_version='py38'
)

predictor = model.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1,
    endpoint_name='plant-classifier-endpoint'
)
```

---

### 🧪 4. `invoke_endpoint.py`

```python
# test/invoke_endpoint.py
import boto3

runtime = boto3.client('sagemaker-runtime')
endpoint_name = 'plant-classifier-endpoint'

with open('sunflower.jpg', 'rb') as f:
    payload = f.read()

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/x-image',
    Body=payload
)

print(response['Body'].read().decode())
```

---

### 📦 `requirements.txt`

```txt
torch==1.12.0
torchvision==0.13.0
sagemaker
boto3
Pillow
```

---

### 📘 README.md Snippet

```markdown
# Plant Classifier on SageMaker

Deploy a PyTorch-based image classifier (daisy 🌼, rose 🌹, sunflower 🌻) to Amazon SageMaker.

## Steps

1. Train model and export to TorchScript.
2. Package with inference.py.
3. Upload to S3 and deploy with `deploy/deploy.py`.
4. Test with `test/invoke_endpoint.py`.

## Prerequisites
- AWS CLI configured
- SageMaker execution role
- IAM permissions for S3, SageMaker, and CloudWatch
```

---

### 📦 Want This as a GitHub Repo?

I can generate a ZIP or GitHub-compatible folder with all of this included — just say:

> ✅ “Send me the ZIP”
> or
> ✅ “Push to GitHub” (and give me a repo name or your GitHub username if you'd like it customized)

Let me know your preference!

 
 **real-time document classification** with **three ML model versions** hosted **outside of SageMaker**

---

### 🧩 **Requirements Recap (No SageMaker):**

* Classify new documents **every 3 seconds**.
* Use **3 versions of a trained ML model** for prediction.
* Models must run with **low latency**.
* Aim for **low operational overhead**.
* Documents arrive in **Amazon S3**.

---

### ✅ **Alternative Architecture (Without SageMaker)**

#### 1. **Model Hosting Options**:

You can host the 3 models on:

* **Amazon ECS / Fargate** – Serverless container hosting.
* **AWS Lambda** – If models are small enough and inference is lightweight.
* **Amazon EC2** – If you want full control (but highest overhead).
* **AWS App Runner** – For simpler model API deployment from container images.

---

#### 2. **Inference API Design**:

* Wrap each model (v1, v2, v3) into a **REST API** using:

  * **Flask / FastAPI** (Python)
  * **Node.js / Express**, etc.
* Host each model version as a **separate API endpoint**, or have a **single API with a version routing parameter** (e.g., `/predict?model=v1`).

---

#### 3. **S3 Trigger and Orchestration**:

* Configure **S3 Event Notification** to invoke a **single AWS Lambda function** upon each new document upload.
* Lambda performs:

  1. Reads the new document from S3.
  2. Sends the content (e.g., text or base64-encoded file) to **each model version API** (3 HTTP calls).
  3. Collects all 3 predictions.
  4. Stores the result (e.g., back in S3, DynamoDB, or sends downstream).

---

### 🔁 **Variant: Using Message Queues**

* If inference takes longer or concurrency grows:

  * **Use Amazon SQS or Amazon EventBridge** instead of direct Lambda processing.
  * Create a **queue-based architecture** where messages trigger processing workers that call APIs.
  * You can scale workers using **AWS Fargate** with **EventBridge Scheduler** or **AWS Lambda with queue triggers**.

---

### 📉 **Comparison to SageMaker Approach**

| Feature                | Without SageMaker         | With SageMaker                     |
| ---------------------- | ------------------------- | ---------------------------------- |
| Hosting Overhead       | Medium (you manage infra) | Low (managed endpoints)            |
| Cost Control           | Cheaper at small scale    | SageMaker incurs higher idle costs |
| Real-time Inference    | Supported                 | Supported                          |
| Model Management       | You manage APIs manually  | Built-in model versioning          |
| Retraining / Pipelines | You build manually        | SageMaker Pipelines available      |
| Scaling                | ECS/Fargate/Lambda        | Auto-scaling via variant weights   |

---

### ✅ **Conclusion**

If you're **not using SageMaker**, the best architecture is:

* Host the 3 models as **REST APIs** on **ECS/Fargate or App Runner**.
* Use a **single Lambda function** triggered by **S3** to:

  * Read the document
  * Call all 3 model APIs
  * Process results
* This offers a **low-latency, scalable, and relatively low-overhead** alternative to SageMaker — with more operational effort, but also more control.

Great — let's break this down clearly.

---

## ✅ What Is a Machine Learning **Model**?

A **machine learning model** is a **mathematical function** trained on data to recognize patterns and make predictions. In simple terms:

> It's like a **smart formula** that can answer questions based on examples it has seen before.

### 🧠 Example:

A model trained to classify plants might answer:

* *"Is this a sunflower, a rose, or a daisy?"*
* Given input features like petal length, color, or an image of the plant.

---

## 🧱 How Is a Model Written?

A model is **not hand-written like rules or if-else logic** — it is **trained** using algorithms and data. But here’s how the process looks in code:

---

### ✍️ 1. **Define the Model Architecture**

You describe the structure — like layers in a neural network, or a decision tree depth.

#### Example (PyTorch – image classification):

```python
import torch.nn as nn

class PlantClassifier(nn.Module):
    def __init__(self):
        super(PlantClassifier, self).__init__()
        self.layer1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 3)  # 3 plant classes

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
```

Or in **scikit-learn** (tabular data example):

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
```

---

### 🔁 2. **Train the Model**

You feed it labeled data so it can **learn patterns**.

```python
model.fit(X_train, y_train)  # scikit-learn example
```

Or in deep learning:

```python
for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

### 💾 3. **Save the Model**

Once trained, the model is saved to disk so it can be reused.

```python
import joblib
joblib.dump(model, 'model_v1.pkl')  # scikit-learn

# PyTorch
torch.save(model.state_dict(), 'model_v1.pt')
```

---

### 🚀 4. **Deploy the Model**

You wrap it in a **web service** or inference engine (e.g., Flask, FastAPI, Lambda, SageMaker endpoint) to accept inputs and return predictions.

```python
@app.post("/predict")
def predict(input_data):
    x = preprocess(input_data)
    prediction = model.predict(x)
    return {"class": prediction}
```

---

## 🔍 Summary:

| Step      | What You Do                                     |
| --------- | ----------------------------------------------- |
| 1. Define | Write model architecture (layers, trees, etc.)  |
| 2. Train  | Use data to teach the model                     |
| 3. Save   | Serialize it to disk                            |
| 4. Serve  | Deploy as an API or service to make predictions |

---

Would you like an example with images (e.g., CNN for plants), or text classification? I can give you full working code or containerize it too.

Great! Let's walk through a **complete example** of how a **machine learning model for image classification** (e.g., plant identification) is **written, trained, saved, and deployed** — using **PyTorch** and **FastAPI**.

---

## 🌿 Use Case: Plant Image Classifier (Sunflower 🌻, Rose 🌹, Daisy 🌼)

---

### 🔧 **1. Define and Train the Model (PyTorch)**

```python
# plant_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Simple CNN model
class PlantCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(PlantCNN, self).__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

# Data loaders (example: folders with 'sunflower', 'rose', 'daisy' subfolders)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_data = datasets.ImageFolder("data/train", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):  # Simplified
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "plant_model.pt")
```

---

### 🚀 **2. Deploy with FastAPI**

```python
# app.py
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
import uvicorn

# Load model
class PlantCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(PlantCNN, self).__init__()
        self.base = models.resnet18(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

model = PlantCNN()
model.load_state_dict(torch.load("plant_model.pt", map_location="cpu"))
model.eval()

# Inference transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Class labels
labels = ['daisy', 'rose', 'sunflower']

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        predicted_class = labels[pred.item()]
    
    return {"class": predicted_class}
```

---

### 🧪 **3. Test the API (with `curl` or Postman)**

```bash
uvicorn app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@sunflower.jpg"
```

Response:

```json
{"class": "sunflower"}
```

---

## 📦 Optional: Dockerize It

```Dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install torch torchvision fastapi uvicorn pillow

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
docker build -t plant-api .
docker run -p 8080:8080 plant-api
```

---

## ✅ Summary

| Step   | What You Did                    |
| ------ | ------------------------------- |
| Model  | Built a CNN using ResNet18      |
| Train  | Trained on labeled plant images |
| Save   | Saved model as `plant_model.pt` |
| Deploy | Served predictions via FastAPI  |
| Test   | Used image upload to classify   |

---

Great! Let’s extend the **Plant Image Classifier** for **AWS deployment** — covering **three deployment options**:

---

## 🚀 **1. Deploy to AWS Lambda (with Amazon API Gateway)**

### ✅ Best For:

* Lightweight models (small `.pt` file)
* Low latency and low traffic
* Serverless, auto-scaling, pay-per-inference

### 📦 Setup Steps:

#### A. Convert to TorchScript (required for Lambda portability)

```python
# convert_model.py
import torch
from plant_classifier import PlantCNN

model = PlantCNN()
model.load_state_dict(torch.load("plant_model.pt"))
model.eval()

scripted_model = torch.jit.script(model)
scripted_model.save("plant_model_scripted.pt")
```

#### B. Create Lambda handler

```python
# lambda_function.py
import json
import base64
import torch
from PIL import Image
from io import BytesIO
from torchvision import transforms
from plant_classifier import PlantCNN  # or load TorchScript

labels = ['daisy', 'rose', 'sunflower']

model = torch.jit.load("plant_model_scripted.pt")
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def lambda_handler(event, context):
    body = json.loads(event['body'])
    img_b64 = body['image']  # base64 encoded
    img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)

    return {
        'statusCode': 200,
        'body': json.dumps({'class': labels[pred.item()]})
    }
```

#### C. Package and Deploy

```bash
zip -r plant-lambda.zip lambda_function.py plant_model_scripted.pt
aws lambda create-function \
  --function-name plantClassifier \
  --zip-file fileb://plant-lambda.zip \
  --handler lambda_function.lambda_handler \
  --runtime python3.9 \
  --role <IAM_ROLE_ARN>
```

Add **API Gateway** to trigger it.

---

## 🐳 **2. Deploy to Amazon ECS (Fargate) using Docker**

### ✅ Best For:

* Containerized inference
* Moderate traffic
* Full control of environment

### 🧱 Steps:

1. Use the same `Dockerfile` from earlier:

```Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install torch torchvision fastapi uvicorn pillow

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

2. Build and push to Amazon ECR:

```bash
aws ecr create-repository --repository-name plant-classifier
docker tag plant-api:latest <aws_account_id>.dkr.ecr.<region>.amazonaws.com/plant-classifier
aws ecr get-login-password | docker login --username AWS --password-stdin <repo-url>
docker push <repo-url>
```

3. Deploy using **AWS Fargate**:

* Create **ECS Task Definition** with ECR image.
* Run behind an **Application Load Balancer**.
* Auto-scale using **CloudWatch metrics**.

---

## 🧠 **3. Deploy on Amazon SageMaker (Real-time Endpoint)**

### ✅ Best For:

* Managed ML inference
* Real-time predictions at scale
* Versioning, auto-scaling, monitoring

---

### 🔧 Steps:

#### A. Create Inference Script

```python
# inference.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

labels = ['daisy', 'rose', 'sunflower']

def model_fn(model_dir):
    model = torch.jit.load(f"{model_dir}/plant_model_scripted.pt")
    model.eval()
    return model

def input_fn(request_body, content_type):
    image = Image.open(io.BytesIO(request_body)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def predict_fn(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs, 1)
    return labels[predicted.item()]

def output_fn(prediction, content_type):
    return prediction
```

#### B. Package Model + Script

```bash
tar -czvf model.tar.gz inference.py plant_model_scripted.pt
```

#### C. Upload to S3 and Create SageMaker Endpoint

```bash
aws s3 cp model.tar.gz s3://<bucket-name>/models/plant-classifier/

from sagemaker.pytorch import PyTorchModel
model = PyTorchModel(
    model_data='s3://<bucket>/models/plant-classifier/model.tar.gz',
    role='<IAM_ROLE>',
    framework_version='1.12.0',
    entry_point='inference.py',
    source_dir='.'
)

predictor = model.deploy(instance_type='ml.m5.large', initial_instance_count=1)
```

#### D. Invoke Endpoint

```python
with open("sunflower.jpg", "rb") as f:
    payload = f.read()

response = predictor.predict(payload)
print(response)
```

---

## 🧩 Comparison Summary

| Deployment Option        | Best Use Case                | Cost   | Auto-Scaling | Complexity         |
| ------------------------ | ---------------------------- | ------ | ------------ | ------------------ |
| **Lambda + API Gateway** | Lightweight, low traffic     | Lowest | Yes          | Low                |
| **ECS/Fargate**          | Medium traffic, full control | Medium | Yes          | Moderate           |
| **SageMaker**            | Real-time ML at scale        | Higher | Yes          | ML-native, managed |

---
✅ Your containerized deployment package (for ECS, Lambda containers, or Docker local) is ready:

👉 [Download plant-classifier-container.zip](sandbox:/mnt/data/plant-classifier-container.zip)

### 📦 What's Included:

* `Dockerfile`: Defines the container image.
* `app.py`: FastAPI-based image classifier server.
* `requirements.txt`: Python dependencies.
* `README.md`: Instructions for Docker and ECS/Fargate deployment.
* `plant_model_scripted.pt`: Placeholder for your TorchScript model file.





